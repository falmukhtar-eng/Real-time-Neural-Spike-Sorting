"""
Spike-Specific Evidential Regression (SSER) - Novel Component #2

Implements evidential regression with class-conditioned hierarchical priors.
Predicts Normal-Inverse-Gamma distribution parameters in one forward pass.

Reference: TEMPUS paper, Section III.C.2
"""

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class EvidentialHead(layers.Layer):
    """
    Evidential Regression Head for morphology prediction.
    
    Outputs parameters of Normal-Inverse-Gamma (NIG) distribution:
        μ: predicted mean (47 PCA coeffs)
        σ²: epistemic uncertainty
        ν: degrees of freedom (aleatoric)
        α, β: Gamma prior parameters
    
    Uses class-conditioned hierarchical priors (novel).
    """
    
    def __init__(self, output_dim=47, num_classes=12, prior_strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.prior_strength = prior_strength
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        
        # Shared transformation
        self.W_share = self.add_weight(
            name='W_share',
            shape=(hidden_dim, hidden_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_share = self.add_weight(
            name='b_share',
            shape=(hidden_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Evidential output heads
        self.W_mu = self.add_weight(
            name='W_mu',
            shape=(hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
        self.W_sigma = self.add_weight(
            name='W_sigma',
            shape=(hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
        self.W_nu = self.add_weight(
            name='W_nu',
            shape=(hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_nu = self.add_weight(
            name='b_nu',
            shape=(self.output_dim,),
            initializer='zeros',
            trainable=True
        )
        
        # Class-conditioned prior parameters (novel)
        self.W_class_alpha = self.add_weight(
            name='W_class_alpha',
            shape=(self.num_classes, hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_class_alpha = self.add_weight(
            name='b_class_alpha',
            shape=(self.num_classes, self.output_dim),
            initializer='zeros',
            trainable=True
        )
        
        self.W_class_beta = self.add_weight(
            name='W_class_beta',
            shape=(self.num_classes, hidden_dim, self.output_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_class_beta = self.add_weight(
            name='b_class_beta',
            shape=(self.num_classes, self.output_dim),
            initializer='zeros',
            trainable=True
        )
        
        super().build(input_shape)
    
    def call(self, inputs, class_logits, return_all=False):
        """
        Forward pass for SSER.
        
        Args:
            inputs: Shared representation [batch, hidden_dim]
            class_logits: Classification logits [batch, num_classes]
            return_all: Return all distribution parameters
        
        Returns:
            mu: Predicted mean [batch, output_dim]
            var: Predictive variance [batch, output_dim]
            (optional) alpha, beta, nu
        """
        batch_size = tf.shape(inputs)[0]
        
        # Shared transformation
        z = tf.nn.relu(tf.matmul(inputs, self.W_share) + self.b_share)
        
        # Get predicted class
        class_probs = tf.nn.softmax(class_logits)
        pred_class = tf.argmax(class_probs, axis=1)
        
        # Evidential parameters
        mu = tf.matmul(z, self.W_mu) + self.b_mu
        sigma_sq = tf.nn.softplus(tf.matmul(z, self.W_sigma) + self.b_sigma)
        nu = tf.nn.softplus(tf.matmul(z, self.W_nu) + self.b_nu) + 1.0
        
        # Class-conditioned prior (novel)
        # Gather prior parameters for predicted class
        indices = tf.stack([tf.range(batch_size), pred_class], axis=1)
        
        # Alpha prior
        alpha_prior = tf.gather_nd(self.W_class_alpha, indices)
        alpha_prior = tf.einsum('bij,bj->bi', alpha_prior, z) + tf.gather_nd(self.b_class_alpha, indices)
        alpha_prior = tf.nn.softplus(alpha_prior) * self.prior_strength
        
        # Beta prior
        beta_prior = tf.gather_nd(self.W_class_beta, indices)
        beta_prior = tf.einsum('bij,bj->bi', beta_prior, z) + tf.gather_nd(self.b_class_beta, indices)
        beta_prior = tf.exp(beta_prior) * self.prior_strength
        
        # Bayesian update: posterior = prior + evidence
        alpha = alpha_prior + 0.5
        beta = beta_prior + 0.5 * sigma_sq * nu
        
        # Predictive variance (closed form, no sampling needed)
        var = sigma_sq * (1.0 + 1.0 / nu) + beta / (alpha - 1.0)
        
        if return_all:
            return mu, var, alpha, beta, nu
        
        return mu, var


class EvidentialLoss:
    """
    Evidential loss function for SSER.
    
    Combines negative log-likelihood (NLL) with evidence regularization.
    """
    
    def __init__(self, lambda_reg=0.01):
        self.lambda_reg = lambda_reg
    
    def nll_loss(self, y_true, mu, nu, alpha, beta):
        """
        Negative log-likelihood under NIG distribution.
        """
        # Student-t distribution approximation
        diff = tf.abs(y_true - mu)
        loss_nll = 0.5 * tf.math.log(np.pi) - tf.math.lgamma(alpha)
        loss_nll += alpha * tf.math.log(beta)
        loss_nll -= (alpha + 0.5) * tf.math.log(beta + 0.5 * nu * diff**2)
        loss_nll += 0.5 * tf.math.log(nu)
        return -tf.reduce_mean(loss_nll)
    
    def evidence_reg(self, alpha, beta, y_true, mu):
        """
        Regularize evidence: penalize high uncertainty on correct predictions.
        """
        diff = tf.abs(y_true - mu)
        # Prediction error should be small when evidence is high
        reg = tf.reduce_mean(diff * beta / (alpha - 1.0))
        return reg
    
    def __call__(self, y_true, mu, var, alpha, beta, nu):
        """
        Compute total evidential loss.
        """
        nll = self.nll_loss(y_true, mu, nu, alpha, beta)
        evidence = self.evidence_reg(alpha, beta, y_true, mu)
        return nll + self.lambda_reg * evidence


class SSERHead(layers.Layer):
    """
    Complete SSER Head combining classification and evidential regression.
    """
    
    def __init__(self, num_classes, output_dim=47, prior_strength=1.0, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.prior_strength = prior_strength
        
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        
        # Classification head
        self.W_cls = self.add_weight(
            name='W_cls',
            shape=(hidden_dim, self.num_classes),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b_cls = self.add_weight(
            name='b_cls',
            shape=(self.num_classes,),
            initializer='zeros',
            trainable=True
        )
        
        # Evidential regression head
        self.evidential = EvidentialHead(
            output_dim=self.output_dim,
            num_classes=self.num_classes,
            prior_strength=self.prior_strength
        )
        
        super().build(input_shape)
    
    def call(self, inputs, return_uncertainty=False):
        """
        Forward pass.
        
        Args:
            inputs: Shared representation [batch, hidden_dim]
            return_uncertainty: Whether to return uncertainty estimates
        
        Returns:
            class_logits: [batch, num_classes]
            morphology: [batch, output_dim] if return_uncertainty=False
            (mu, var, alpha, beta, nu) if return_uncertainty=True
        """
        # Classification
        class_logits = tf.matmul(inputs, self.W_cls) + self.b_cls
        
        # Evidential regression
        if return_uncertainty:
            mu, var, alpha, beta, nu = self.evidential(inputs, class_logits, return_all=True)
            return class_logits, mu, var, alpha, beta, nu
        else:
            mu, _ = self.evidential(inputs, class_logits, return_all=False)
            return class_logits, mu
