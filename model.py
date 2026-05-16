"""
CNN-BiLSTM Model for Real-time Neural Spike Sorting and Morphology Forecasting
Implements Algorithm 1 (Section III.C) and Table IV architecture exactly.

Paper: "Real-time Neural Spike Sorting and Morphology Forecasting Using Hybrid 
        CNN-BiLSTM Architectures for Brain-Computer Interface Applications"
Author: Dr. Firas Almukhtar

Architecture Summary (from Table IV):
- Input: 47 PCA-reduced spike waveform samples
- CNN Stage: 3 Conv1D layers (64→128→256 filters, kernel=3) + ELU + AvgPool
- BiLSTM Stage: 2 bidirectional LSTM layers (128 units each)
- Dual Output: Classification (Softmax, K classes) + Morphology Prediction (Linear, 47 values)
- Total parameters: 952,507
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


class CNNBiLSTMSpikeSorter(keras.Model):
    """
    Hybrid CNN-BiLSTM for simultaneous spike sorting and morphology prediction.

    Key design choices from paper:
    - Sequence length: 16 consecutive spikes (Section III.C.2)
    - Zero-padding for edge cases (first 15 and last spikes)
    - Prediction is single-step: given sequence up to spike t, predicts spike t+1
    - Prediction invalid if ISI > 500ms (Section III.C.3)
    - Values clipped to [-5, +5] standard deviations during training
    """

    def __init__(
        self,
        num_classes=8,           # K neuron classes (4-12 depending on dataset, Table I)
        num_pca_components=47,    # From scree plot analysis (Section III.B.2)
        sequence_length=16,       # Fixed-length sequence (Section III.C.2)
        cnn_filters=[64, 128, 256],  # Table IV: Layers 1, 4, 7
        lstm_units=128,           # Table IV: Layers 11, 13
        dropout_rates=[0.3, 0.2, 0.2],  # Table IV: Layers 12, 14-prep, 15
        prediction_clip=5.0,      # [-5, +5] sigma clipping (Algorithm 1)
        name="cnn_bilstm_spike_sorter",
        **kwargs
    ):
        super(CNNBiLSTMSpikeSorter, self).__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.num_pca_components = num_pca_components
        self.sequence_length = sequence_length
        self.prediction_clip = prediction_clip

        # === STAGE 1: Spatial Feature Extraction (CNN Layers 1-9) ===
        # Layer 1: Conv1D, 64 filters, kernel=3
        self.conv1 = layers.Conv1D(
            filters=cnn_filters[0], 
            kernel_size=3, 
            strides=1,
            padding='valid',
            name='conv1d_1'
        )
        # Layer 2: ELU activation
        self.elu1 = layers.Activation('elu', name='elu_1')
        # Layer 3: Average Pooling, pool=2
        self.pool1 = layers.AveragePooling1D(pool_size=2, name='avg_pool_1')

        # Layer 4: Conv1D, 128 filters, kernel=3
        self.conv2 = layers.Conv1D(
            filters=cnn_filters[1], 
            kernel_size=3, 
            strides=1,
            padding='valid',
            name='conv1d_2'
        )
        # Layer 5: ELU activation
        self.elu2 = layers.Activation('elu', name='elu_2')
        # Layer 6: Average Pooling, pool=2
        self.pool2 = layers.AveragePooling1D(pool_size=2, name='avg_pool_2')

        # Layer 7: Conv1D, 256 filters, kernel=3
        self.conv3 = layers.Conv1D(
            filters=cnn_filters[2], 
            kernel_size=3, 
            strides=1,
            padding='valid',
            name='conv1d_3'
        )
        # Layer 8: ELU activation
        self.elu3 = layers.Activation('elu', name='elu_3')
        # Layer 9: Global Average Pooling → 256-dim vector
        self.global_pool = layers.GlobalAveragePooling1D(name='global_avg_pool')

        # === STAGE 2: Temporal Modeling (BiLSTM Layers 10-13) ===
        # Layer 10: Reshape for LSTM input (implicit in Keras)
        # Layer 11: BiLSTM (1st), 128 units × 2
        self.bilstm1 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, name='lstm_1'),
            name='bilstm_1'
        )
        # Layer 12: Dropout 30%
        self.dropout1 = layers.Dropout(dropout_rates[0], name='dropout_1')

        # Layer 13: BiLSTM (2nd), 128 units × 2
        self.bilstm2 = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=False, name='lstm_2'),
            name='bilstm_2'
        )
        # Layer 14-prep: Dropout 20%
        self.dropout2 = layers.Dropout(dropout_rates[1], name='dropout_2')

        # === STAGE 3: Dual-Output Generation (Layers 14-16) ===
        # Layer 14: Dense, 128 units, ReLU
        self.dense_shared = layers.Dense(128, activation='relu', name='dense_shared')
        # Layer 15: Dropout 20%
        self.dropout3 = layers.Dropout(dropout_rates[2], name='dropout_3')

        # Output 16a: Classification (Softmax)
        self.classification_head = layers.Dense(
            num_classes, 
            activation='softmax', 
            name='classification_output'
        )

        # Output 16b: Morphology Prediction (Linear regression, 47 PCA coefficients)
        self.prediction_head = layers.Dense(
            num_pca_components, 
            activation='linear', 
            name='morphology_prediction_output'
        )

    def call(self, inputs, training=False):
        """
        Forward pass implementing Algorithm 1.

        Args:
            inputs: Tensor of shape (batch, sequence_length, num_pca_components)
                   Each element is PCA-reduced spike waveform (47 components)
            training: Boolean, whether in training mode (for dropout)

        Returns:
            classification_output: (batch, num_classes) - Softmax probabilities
            morphology_prediction: (batch, num_pca_components) - Predicted PCA coeffs
        """
        # inputs shape: (batch, seq_len, 47)
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Process each spike in the sequence through CNN (shared weights)
        # Reshape to process all spikes: (batch*seq_len, 47, 1)
        x = tf.reshape(inputs, [-1, self.num_pca_components, 1])

        # CNN Stage (Layers 1-9)
        x = self.conv1(x)        # (batch*seq_len, 45, 64)
        x = self.elu1(x)
        x = self.pool1(x)        # (batch*seq_len, 22, 64)

        x = self.conv2(x)        # (batch*seq_len, 20, 128)
        x = self.elu2(x)
        x = self.pool2(x)        # (batch*seq_len, 10, 128)

        x = self.conv3(x)        # (batch*seq_len, 8, 256)
        x = self.elu3(x)
        x = self.global_pool(x)  # (batch*seq_len, 256)

        # Reshape back to sequence: (batch, seq_len, 256)
        x = tf.reshape(x, [batch_size, seq_len, 256])

        # BiLSTM Stage (Layers 10-13)
        x = self.bilstm1(x)      # (batch, seq_len, 256)
        x = self.dropout1(x, training=training)
        x = self.bilstm2(x)      # (batch, 256)
        x = self.dropout2(x, training=training)

        # Shared dense (Layer 14-15)
        x = self.dense_shared(x)  # (batch, 128)
        x = self.dropout3(x, training=training)

        # Dual outputs
        classification = self.classification_head(x)  # (batch, num_classes)
        morphology_pred = self.prediction_head(x)     # (batch, 47)

        # Clip prediction to [-5, +5] sigma during training (Algorithm 1)
        if training:
            morphology_pred = tf.clip_by_value(
                morphology_pred, 
                -self.prediction_clip, 
                self.prediction_clip
            )

        return classification, morphology_pred

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'num_pca_components': self.num_pca_components,
            'sequence_length': self.sequence_length,
            'prediction_clip': self.prediction_clip,
        })
        return config


def build_model(num_classes=8, num_pca_components=47, sequence_length=16):
    """
    Build and return compiled CNN-BiLSTM model.

    Total trainable parameters: ~952,507 (Table IV)
    """
    model = CNNBiLSTMSpikeSorter(
        num_classes=num_classes,
        num_pca_components=num_pca_components,
        sequence_length=sequence_length
    )

    # Build by calling once
    dummy_input = tf.zeros((1, sequence_length, num_pca_components))
    _ = model(dummy_input, training=False)

    return model


def verify_parameter_count():
    """Verify total trainable parameters matches Table IV: 952,507"""
    model = build_model(num_classes=8, num_pca_components=47, sequence_length=16)
    model.summary()

    total_params = model.count_params()
    expected_params = 952507

    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Expected (Table IV): {expected_params:,}")
    print(f"Match: {'YES' if total_params == expected_params else 'NO - CHECK ARCHITECTURE'}")

    return total_params


if __name__ == "__main__":
    verify_parameter_count()
