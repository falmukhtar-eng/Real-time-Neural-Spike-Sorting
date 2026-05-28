"""
Complete TEMPUS Model

Integrates all novel components:
- Adaptive Spike Attention (ASA)
- Spike-Specific Evidential Regression (SSER)
- CNN-BiLSTM backbone
- Dual output (classification + evidential morphology)

Reference: TEMPUS paper, Section III
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

# Import TEMPUS custom components
from models.attention import AdaptiveSpikeAttention
from models.evidential import SSERHead
from models.losses import TEMPUSCombinedLoss


class CNNBackbone(layers.Layer):
    """
    CNN feature extractor for spike waveforms.
    """
    
    def __init__(self, filters=[64, 128, 256], kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.convs = []
        self.activations = []
        self.pools = []
        
        for i, f in enumerate(filters):
            self.convs.append(
                layers.Conv1D(f, kernel_size, padding='causal', name=f'conv_{i}')
            )
            self.activations.append(layers.ELU(alpha=1.0, name=f'elu_{i}'))
            self.pools.append(layers.AvgPool1D(pool_size=2, name=f'pool_{i}'))
    
    def call(self, inputs):
        x = inputs
        for conv, act, pool in zip(self.convs, self.activations, self.pools):
            x = conv(x)
            x = act(x)
            x = pool(x)
        
        # Global average pooling
        x = tf.reduce_mean(x, axis=1)
        return x


class BiLSTMBackbone(layers.Layer):
    """
    BiLSTM temporal modeler with ASA integration.
    """
    
    def __init__(self, units=128, return_sequences=False, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout = dropout
        
        self.lstm = layers.Bidirectional(
            layers.LSTM(units, return_sequences=return_sequences, dropout=dropout),
            name='bilstm'
        )
    
    def call(self, inputs):
        return self.lstm(inputs)


class TEMPUS(Model):
    """
    Complete TEMPUS Algorithm.
    
    Integrates:
    1. CNN for spatial feature extraction
    2. BiLSTM for temporal modeling
    3. Adaptive Spike Attention (ASA) - Novel
    4. Spike-Specific Evidential Regression (SSER) - Novel
    """
    
    def __init__(self,
                 num_classes=12,
                 pca_dim=47,
                 lstm_units=128,
                 lstm_layers=2,
                 cnn_filters=[64, 128, 256],
                 kernel_size=3,
                 attention_window=16,
                 attention_sensitivity=0.5,
                 reference_rate=50.0,
                 prior_strength=1.0,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.num_classes = num_classes
        self.pca_dim = pca_dim
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        
        # Feature extraction
        self.cnn_backbone = CNNBackbone(filters=cnn_filters, kernel_size=kernel_size)
        
        # Temporal modeling with multiple BiLSTM layers
        self.bilstm_layers = []
        for i in range(lstm_layers):
            self.bilstm_layers.append(
                BiLSTMBackbone(units=lstm_units, 
                               return_sequences=(i < lstm_layers - 1),
                               name=f'bilstm_{i}')
            )
        
        # Adaptive Spike Attention (Novel Component #1)
        self.asa = AdaptiveSpikeAttention(
            units=lstm_units * 2,  # Bidirectional output
            base_window=attention_window,
            rate_sensitivity=attention_sensitivity,
            reference_rate=reference_rate,
            name='asa'
        )
        
        # SSER Head (Novel Component #2)
        self.sser_head = SSERHead(
            num_classes=num_classes,
            output_dim=pca_dim,
            prior_strength=prior_strength,
            name='sser'
        )
        
        # Dropout for regularization
        self.dropout = layers.Dropout(0.2)
    
    def encode(self, spikes, spike_times=None):
        """
        Encode spikes into embeddings (for TCR).
        
        Args:
            spikes: [batch, seq_len, pca_dim] PCA-reduced spikes
            spike_times: Optional spike timestamps for ASA
        
        Returns:
            Embeddings [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, pca_dim = spikes.shape
        
        # Process each spike through CNN
        spikes_flat = tf.reshape(spikes, [-1, pca_dim])
        cnn_out = self.cnn_backbone(tf.expand_dims(spikes_flat, axis=-1))
        cnn_out = tf.reshape(cnn_out, [batch_size, seq_len, -1])
        
        # BiLSTM layers
        x = cnn_out
        for bilstm in self.bilstm_layers[:-1]:  # All except last
            x = bilstm(x)
        
        # Apply ASA (Novel)
        if spike_times is not None:
            x = self.asa(x, spike_times=spike_times)
        else:
            x = self.asa(x)
        
        # Final BiLSTM
        x = self.bilstm_layers[-1](x)
        
        return self.dropout(x)
    
    def call(self, spikes, spike_times=None, return_uncertainty=False, training=False):
        """
        Forward pass.
        
        Args:
            spikes: [batch, seq_len, pca_dim] PCA-reduced spikes
            spike_times: Optional spike timestamps [batch, seq_len]
            return_uncertainty: Whether to return uncertainty estimates
            training: Training mode flag
        
        Returns:
            If return_uncertainty=False:
                class_logits, morphology_mean
            If return_uncertainty=True:
                class_logits, morphology_mean, morphology_var, alpha, beta, nu
        """
        batch_size, seq_len, pca_dim = spikes.shape
        
        # Process each spike through CNN
        spikes_flat = tf.reshape(spikes, [-1, pca_dim])
        cnn_out = self.cnn_backbone(tf.expand_dims(spikes_flat, axis=-1))
        cnn_out = tf.reshape(cnn_out, [batch_size, seq_len, -1])
        
        # BiLSTM layers with ASA between them
        x = cnn_out
        
        # First BiLSTM (returns sequences)
        x = self.bilstm_layers[0](x)
        
        # Adaptive Spike Attention (Novel)
        if spike_times is not None:
            x = self.asa(x, spike_times=spike_times)
        else:
            x = self.asa(x)
        
        # Second BiLSTM (returns last output)
        x = self.bilstm_layers[1](x)
        x = self.dropout(x, training=training)
        
        # SSER Head (Novel)
        if return_uncertainty:
            class_logits, mu, var, alpha, beta, nu = self.sser_head(
                x, return_uncertainty=True
            )
            return class_logits, mu, var, alpha, beta, nu
        else:
            class_logits, mu = self.sser_head(x, return_uncertainty=False)
            return class_logits, mu
    
    def predict_with_uncertainty(self, spikes, spike_times=None):
        """
        Predict morphology with uncertainty (single forward pass).
        No sampling needed - SSER provides closed-form uncertainty.
        """
        class_logits, mu, var, alpha, beta, nu = self(
            spikes, spike_times=spike_times, return_uncertainty=True
        )
        
        # Total predictive variance
        total_var = var
        
        # Classification confidence
        class_probs = tf.nn.softmax(class_logits)
        confidence = tf.reduce_max(class_probs, axis=-1)
        
        return {
            'class_logits': class_logits,
            'class_probs': class_probs,
            'confidence': confidence,
            'morphology_mean': mu,
            'morphology_var': total_var,
            'evidential_alpha': alpha,
            'evidential_beta': beta,
            'evidential_nu': nu
        }
    
    def reject_by_confidence(self, predictions, confidence_threshold=0.9):
        """
        Clinical rejection based on confidence threshold.
        """
        accept_mask = predictions['confidence'] >= confidence_threshold
        return accept_mask


def create_tempus_model(num_classes=12,
                        pca_dim=47,
                        lstm_units=128,
                        lstm_layers=2,
                        cnn_filters=[64, 128, 256],
                        kernel_size=3,
                        attention_window=16,
                        attention_sensitivity=0.5,
                        reference_rate=50.0,
                        prior_strength=1.0,
                        learning_rate=0.001):
    """
    Factory function to create and compile TEMPUS model.
    """
    model = TEMPUS(
        num_classes=num_classes,
        pca_dim=pca_dim,
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        cnn_filters=cnn_filters,
        kernel_size=kernel_size,
        attention_window=attention_window,
        attention_sensitivity=attention_sensitivity,
        reference_rate=reference_rate,
        prior_strength=prior_strength
    )
    
    # Compile with optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss=TEMPUSCombinedLoss(),
        metrics=['accuracy']
    )
    
    return model
