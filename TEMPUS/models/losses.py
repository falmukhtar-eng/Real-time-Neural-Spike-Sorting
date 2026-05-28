"""
Novel Loss Functions for TEMPUS

1. Physiologically-Constrained Loss (PCL) - Novel Component #3
2. Temporal Consistency Regularization (TCR) - Novel Component #4

Reference: TEMPUS paper, Sections III.D.1 and III.D.2
"""

import tensorflow as tf
import numpy as np


class PhysiologicallyConstrainedLoss:
    """
    Physiologically-Constrained Loss (PCL)
    
    Enforces neurophysiological constraints on predicted waveforms:
    - Peak negativity (peak should be between -50 and -20 μV)
    - Peak width (FWHM between 4-12 samples at 24 kHz)
    - Repolarization slope (negative after peak)
    - Refractory period (no secondary peak within 1 ms)
    
    Reference: TEMPUS paper, Section III.D.2
    """
    
    def __init__(self,
                 sampling_rate=24000,
                 lambda_peak=10.0,
                 lambda_width=5.0,
                 lambda_slope=5.0,
                 lambda_ref=10.0):
        self.sampling_rate = sampling_rate
        self.lambda_peak = lambda_peak
        self.lambda_width = lambda_width
        self.lambda_slope = lambda_slope
        self.lambda_ref = lambda_ref
        
        # Physiological bounds
        self.peak_min = -50.0   # μV (most negative)
        self.peak_max = -20.0   # μV (least negative)
        self.width_min = 4      # samples at 24 kHz (0.17 ms)
        self.width_max = 12     # samples (0.5 ms)
        self.refractory_min = int(1.0 * sampling_rate / 1000)  # 24 samples at 24 kHz
    
    def _extract_waveform_metrics(self, waveform):
        """
        Extract physiological metrics from waveform.
        
        Args:
            waveform: [batch, 200] predicted waveform in μV
        
        Returns:
            Dictionary of metrics
        """
        batch_size = tf.shape(waveform)[0]
        
        # Find peak (most negative sample)
        peak_values = tf.reduce_min(waveform, axis=1)
        peak_indices = tf.argmin(waveform, axis=1)
        
        # Compute Full Width at Half Maximum (FWHM)
        def compute_fwhm(w, peak_idx):
            half_max = (tf.reduce_min(w) + tf.reduce_mean(w)) / 2.0
            # Find left crossing
            left = peak_idx
            for i in range(peak_idx - 1, -1, -1):
                if w[i] > half_max:
                    left = i
                    break
            # Find right crossing
            right = peak_idx
            for i in range(peak_idx + 1, tf.shape(w)[0]):
                if w[i] > half_max:
                    right = i
                    break
            return tf.cast(right - left, tf.float32)
        
        fwhm = tf.map_fn(
            lambda x: compute_fwhm(x[0], x[1]),
            (waveform, peak_indices),
            dtype=tf.float32
        )
        
        # Compute repolarization slope (after peak)
        def compute_slope(w, peak_idx):
            end_idx = tf.minimum(peak_idx + 8, tf.shape(w)[0])
            slope = (w[end_idx - 1] - w[peak_idx]) / tf.cast(end_idx - peak_idx, tf.float32)
            return slope
        
        repolarization_slope = tf.map_fn(
            lambda x: compute_slope(x[0], x[1]),
            (waveform, peak_indices),
            dtype=tf.float32
        )
        
        # Detect refractory violation (secondary peak within refractory period)
        def check_refractory(w, peak_idx):
            refractory_end = tf.minimum(peak_idx + self.refractory_min, tf.shape(w)[0])
            post_window = w[peak_idx:refractory_end]
            min_post = tf.reduce_min(post_window)
            # Secondary peak if > 50% of main peak
            return tf.cast(min_post < 0.5 * w[peak_idx], tf.float32)
        
        refractory_violation = tf.map_fn(
            lambda x: check_refractory(x[0], x[1]),
            (waveform, peak_indices),
            dtype=tf.float32
        )
        
        return {
            'peak': peak_values,
            'peak_idx': peak_indices,
            'fwhm': fwhm,
            'repol_slope': repolarization_slope,
            'refractory_violation': refractory_violation
        }
    
    def peak_penalty(self, peak):
        """
        Penalize peaks outside physiological range [-50, -20] μV.
        """
        # Peak should be negative and within range
        too_shallow = tf.nn.relu(peak - self.peak_max)  # peak > -20
        too_deep = tf.nn.relu(self.peak_min - peak)     # peak < -50
        return too_shallow + too_deep
    
    def width_penalty(self, fwhm):
        """
        Penalize FWHM outside [4, 12] samples.
        """
        too_narrow = tf.nn.relu(self.width_min - fwhm)
        too_wide = tf.nn.relu(fwhm - self.width_max)
        return too_narrow + too_wide
    
    def slope_penalty(self, slope):
        """
        Penalize positive repolarization slope.
        """
        # Slope should be negative (repolarizing)
        positive_slope = tf.nn.relu(slope + 0.5)  # Allow slight negative
        return positive_slope
    
    def refractory_penalty(self, violation):
        """
        Penalize refractory period violations.
        """
        return violation * 10.0  # Strong penalty
    
    def mse_loss(self, y_pred, y_true):
        """
        Standard MSE loss.
        """
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    def __call__(self, y_pred_pca, y_true_pca, y_pred_waveform=None, y_true_waveform=None):
        """
        Compute total PCL.
        
        Args:
            y_pred_pca: Predicted PCA coefficients [batch, 47]
            y_true_pca: True PCA coefficients [batch, 47]
            y_pred_waveform: Optional reconstructed waveform [batch, 200]
            y_true_waveform: Optional true waveform [batch, 200]
        
        Returns:
            Total PCL loss
        """
        # MSE loss on PCA coefficients
        loss_mse = self.mse_loss(y_pred_pca, y_true_pca)
        
        # If waveforms not provided, use PCA reconstruction (approximate)
        if y_pred_waveform is None:
            return loss_mse
        
        # Extract physiological metrics
        metrics = self._extract_waveform_metrics(y_pred_waveform)
        
        # Compute penalties
        loss_peak = tf.reduce_mean(self.peak_penalty(metrics['peak']))
        loss_width = tf.reduce_mean(self.width_penalty(metrics['fwhm']))
        loss_slope = tf.reduce_mean(self.slope_penalty(metrics['repol_slope']))
        loss_ref = tf.reduce_mean(self.refractory_penalty(metrics['refractory_violation']))
        
        # Total PCL
        total_loss = (
            loss_mse +
            self.lambda_peak * loss_peak +
            self.lambda_width * loss_width +
            self.lambda_slope * loss_slope +
            self.lambda_ref * loss_ref
        )
        
        return total_loss


class TemporalConsistencyRegularization:
    """
    Temporal Consistency Regularization (TCR)
    
    Enforces that temporally adjacent spikes from the same neuron
    produce similar embeddings. Leverages refractory period dynamics
    for semi-supervised learning.
    
    Reference: TEMPUS paper, Section III.D.1
    """
    
    def __init__(self,
                 refractory_ms=2.0,
                 burst_ms=10.0,
                 sampling_rate=24000,
                 margin=1.0,
                 lambda_ref=10.0,
                 temperature=0.5):
        self.refractory_samples = int(refractory_ms * sampling_rate / 1000)
        self.burst_samples = int(burst_ms * sampling_rate / 1000)
        self.margin = margin
        self.lambda_ref = lambda_ref
        self.temperature = temperature
    
    def compute_embeddings(self, model, spikes):
        """
        Extract embeddings from model for unlabeled spikes.
        
        Args:
            model: TEMPUS model (without classification head)
            spikes: Unlabeled spike waveforms [batch, seq_len, 200]
        
        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        # Forward pass through encoder only
        embeddings = model.encode(spikes)
        return embeddings
    
    def contrastive_loss(self, anchor, positive, negative):
        """
        Standard contrastive loss with temperature scaling.
        """
        # Normalize embeddings
        anchor = tf.nn.l2_normalize(anchor, axis=-1)
        positive = tf.nn.l2_normalize(positive, axis=-1)
        negative = tf.nn.l2_normalize(negative, axis=-1)
        
        # Positive similarity
        pos_sim = tf.reduce_sum(anchor * positive, axis=-1) / self.temperature
        
        # Negative similarity
        neg_sim = tf.reduce_sum(anchor * negative, axis=-1) / self.temperature
        
        # InfoNCE loss
        loss_pos = -tf.reduce_mean(tf.nn.log_softmax(pos_sim))
        loss_neg = tf.reduce_mean(tf.nn.log_softmax(neg_sim))
        
        return loss_pos + loss_neg
    
    def refractory_penalty(self, spike_times, spike_presence):
        """
        Penalize spikes within refractory period.
        """
        # Spike times in samples
        intervals = spike_times[1:] - spike_times[:-1]
        violations = tf.cast(intervals < self.refractory_samples, tf.float32)
        return self.lambda_ref * tf.reduce_mean(violations * spike_presence[1:])
    
    def __call__(self, model, unlabeled_spikes, spike_times, neighbor_indices=None):
        """
        Compute TCR loss.
        
        Args:
            model: TEMPUS model
            unlabeled_spikes: [batch, seq_len, 200]
            spike_times: [batch, seq_len] spike timestamps in samples
            neighbor_indices: Optional pre-computed neighbor indices
        
        Returns:
            Total TCR loss
        """
        # Extract embeddings
        embeddings = model.encode(unlabeled_spikes)
        batch_size, seq_len, hidden_dim = embeddings.shape
        
        # Build positive pairs (temporally close spikes)
        positive_losses = []
        negative_losses = []
        
        for t in range(seq_len):
            # Find temporal neighbors within burst window
            current_time = spike_times[:, t]
            
            # Positive pairs: adjacent spikes in burst
            for dt in [-2, -1, 1, 2]:
                if 0 <= t + dt < seq_len:
                    anchor = embeddings[:, t, :]
                    positive = embeddings[:, t+dt, :]
                    # Random negative from different time
                    neg_idx = tf.random.uniform([batch_size], 0, seq_len, dtype=tf.int32)
                    negative = tf.gather(embeddings, neg_idx, axis=1, batch_dims=1)
                    
                    pos_loss = self.contrastive_loss(anchor, positive, negative)
                    positive_losses.append(pos_loss)
            
            # Negative pairs: random spikes from different sequences
            for _ in range(2):
                neg_idx = tf.random.uniform([batch_size], 0, batch_size, dtype=tf.int32)
                neg_time = tf.random.uniform([batch_size], 0, seq_len, dtype=tf.int32)
                negative = tf.gather(embeddings, neg_time, axis=1, batch_dims=1)
                negative = tf.gather(negative, neg_idx, axis=0)
                anchor = embeddings[:, t, :]
                pos_loss = self.contrastive_loss(anchor, anchor, negative)
                negative_losses.append(pos_loss)
        
        # Refractory penalty
        spike_presence = tf.ones_like(spike_times)
        loss_ref = self.refractory_penalty(tf.reshape(spike_times, [-1]), 
                                           tf.reshape(spike_presence, [-1]))
        
        # Total TCR loss
        loss_pos = tf.reduce_mean(positive_losses) if positive_losses else 0.0
        loss_neg = tf.reduce_mean(negative_losses) if negative_losses else 0.0
        
        return loss_pos + loss_neg + loss_ref


class TEMPUSCombinedLoss:
    """
    Combined loss function for TEMPUS.
    
    L_total = L_supervised + λ₂ * L_TCR + λ₃ * L_entropy
    where L_supervised = L_classification + λ₁ * L_PCL
    """
    
    def __init__(self,
                 pcl_weight=1.0,
                 tcr_weight=0.5,
                 entropy_weight=0.1,
                 **pcl_kwargs):
        self.pcl = PhysiologicallyConstrainedLoss(**pcl_kwargs)
        self.tcr = TemporalConsistencyRegularization()
        self.pcl_weight = pcl_weight
        self.tcr_weight = tcr_weight
        self.entropy_weight = entropy_weight
    
    def classification_loss(self, y_pred, y_true):
        """
        Cross-entropy loss for classification.
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        ))
    
    def entropy_regularization(self, logits):
        """
        Entropy minimization on unlabeled data.
        """
        probs = tf.nn.softmax(logits)
        entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=-1)
        return tf.reduce_mean(entropy)
    
    def __call__(self, predictions, targets, unlabeled_data=None):
        """
        Compute total loss.
        
        Args:
            predictions: Dict containing 'class_logits', 'morphology', etc.
            targets: Dict containing 'class_labels', 'morphology_pca'
            unlabeled_data: Optional tuple (unlabeled_spikes, spike_times)
        
        Returns:
            Total loss dictionary
        """
        # Supervised losses
        loss_cls = self.classification_loss(
            predictions['class_logits'], 
            targets['class_labels']
        )
        
        loss_morph = self.pcl(
            predictions['morphology_pca'],
            targets['morphology_pca'],
            predictions.get('waveform_reconstructed'),
            targets.get('waveform')
        )
        
        loss_supervised = loss_cls + self.pcl_weight * loss_morph
        
        total_loss = loss_supervised
        
        loss_dict = {
            'classification': loss_cls,
            'morphology': loss_morph,
            'supervised': loss_supervised
        }
        
        # Semi-supervised TCR loss
        if unlabeled_data is not None:
            unlabeled_spikes, spike_times = unlabeled_data
            loss_tcr = self.tcr(predictions['model'], unlabeled_spikes, spike_times)
            total_loss += self.tcr_weight * loss_tcr
            loss_dict['tcr'] = loss_tcr
        
        # Entropy regularization on unlabeled data
        if 'unlabeled_logits' in predictions:
            loss_entropy = self.entropy_regularization(predictions['unlabeled_logits'])
            total_loss += self.entropy_weight * loss_entropy
            loss_dict['entropy'] = loss_entropy
        
        loss_dict['total'] = total_loss
        
        return loss_dict
