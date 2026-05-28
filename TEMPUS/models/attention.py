"""
Adaptive Spike Attention (ASA) - Novel Component #1

Implements firing-rate-adaptive attention mechanism where the receptive
field dynamically adjusts based on estimated spike rate.

Reference: TEMPUS paper, Section III.C.1
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class AdaptiveSpikeAttention(layers.Layer):
    """
    Adaptive Spike Attention (ASA) Layer.
    
    Dynamically adjusts attention window based on estimated firing rate.
    During bursts (high rate), window contracts. During sparse firing
    (low rate), window expands.
    
    Args:
        units: Hidden dimension (default: 256)
        base_window: Base attention window in spikes (default: 16)
        rate_sensitivity: Gamma parameter (default: 0.5)
        reference_rate: Reference firing rate in Hz (default: 50.0)
        sampling_rate: Data sampling rate in Hz (default: 24000)
    """
    
    def __init__(self,
                 units=256,
                 base_window=16,
                 rate_sensitivity=0.5,
                 reference_rate=50.0,
                 sampling_rate=24000,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.base_window = base_window
        self.gamma = rate_sensitivity
        self.ref_rate = reference_rate
        self.sampling_rate = sampling_rate
        
        # Learnable attention parameters
        self.W_a = None
        self.v = None
        self.b_a = None
        
    def build(self, input_shape):
        # Attention query projection
        self.W_a = self.add_weight(
            name='W_a',
            shape=(self.units, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_a = self.add_weight(
            name='b_a',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        # Attention score vector
        self.v = self.add_weight(
            name='v',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super().build(input_shape)
    
    def _estimate_firing_rate(self, spike_times, current_idx, window_ms=100):
        """
        Estimate local firing rate from spike times.
        
        Args:
            spike_times: Array of spike timestamps in samples
            current_idx: Current spike index
            window_ms: Estimation window in milliseconds
        
        Returns:
            Estimated firing rate in Hz
        """
        if spike_times is None or len(spike_times) < 2:
            return self.ref_rate
        
        # Get previous spikes within window
        current_time = spike_times[current_idx]
        window_samples = int(window_ms * self.sampling_rate / 1000)
        
        past_spikes = []
        for i in range(max(0, current_idx - 50), current_idx):
            if current_time - spike_times[i] <= window_samples:
                past_spikes.append(spike_times[i])
        
        if len(past_spikes) < 2:
            return self.ref_rate
        
        # Calculate instantaneous rate
        time_span = (current_time - past_spikes[0]) / self.sampling_rate
        if time_span > 0:
            rate = len(past_spikes) / time_span
        else:
            rate = self.ref_rate
        
        # Exponential moving average for smoothing
        if not hasattr(self, '_prev_rate'):
            self._prev_rate = rate
        else:
            rate = 0.7 * rate + 0.3 * self._prev_rate
            self._prev_rate = rate
        
        return min(max(rate, 5.0), 200.0)  # Clamp to [5, 200] Hz
    
    def _compute_adaptive_window(self, firing_rate):
        """
        Compute adaptive window size based on firing rate.
        
        W_t = W_base * sigmoid(gamma * (r_ref - r_hat))
        """
        rate_diff = self.ref_rate - firing_rate
        modulation = tf.sigmoid(self.gamma * rate_diff)
        window_size = self.base_window * modulation
        return tf.cast(tf.math.ceil(window_size), tf.int32)
    
    def call(self, inputs, spike_times=None, return_attention=False):
        """
        Forward pass for ASA.
        
        Args:
            inputs: Hidden states [batch, seq_len, units]
            spike_times: Optional spike timestamps for rate estimation
            return_attention: Whether to return attention weights
        
        Returns:
            Context vector [batch, units]
            Optional: attention weights [batch, seq_len]
        """
        batch_size, seq_len, _ = tf.shape(inputs).numpy()
        
        # Estimate firing rate for each sequence
        if spike_times is not None:
            # Use actual spike times for rate estimation
            firing_rates = []
            for b in range(batch_size):
                rate = self._estimate_firing_rate(spike_times[b], seq_len - 1)
                firing_rates.append(rate)
            firing_rates = tf.constant(firing_rates, dtype=tf.float32)
        else:
            # Fallback: estimate from inter-spike intervals in embeddings
            firing_rates = tf.ones(batch_size) * self.ref_rate
        
        # Compute adaptive window per batch element
        windows = [self._compute_adaptive_window(r) for r in firing_rates]
        max_window = max(windows)
        
        # Pad sequences to max_window if needed
        if max_window > seq_len:
            padding = max_window - seq_len
            inputs = tf.pad(inputs, [[0, 0], [padding, 0], [0, 0]])
        
        # Compute attention scores
        attention_weights = []
        context_vectors = []
        
        for b in range(batch_size):
            window = windows[b]
            seq = inputs[b, -window:, :]  # Take last window spikes
            
            # Project queries
            projected = tf.tanh(tf.matmul(seq, self.W_a) + self.b_a)
            
            # Compute attention scores
            scores = tf.matmul(projected, self.v)
            scores = tf.squeeze(scores, axis=-1)
            
            # Rate modulation (novel: attention scores scaled by rate)
            rate_mod = 1.0 + 0.2 * (firing_rates[b] - self.ref_rate) / self.ref_rate
            scores = scores * rate_mod
            
            # Softmax normalization
            weights = tf.nn.softmax(scores)
            attention_weights.append(weights)
            
            # Weighted sum
            context = tf.reduce_sum(seq * tf.expand_dims(weights, -1), axis=0)
            context_vectors.append(context)
        
        context = tf.stack(context_vectors, axis=0)
        
        if return_attention:
            return context, attention_weights
        
        return context
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'base_window': self.base_window,
            'rate_sensitivity': self.gamma,
            'reference_rate': self.ref_rate,
            'sampling_rate': self.sampling_rate
        })
        return config
