"""
Trainable fuzzy inference layer (TSK / ANFIS-style) for AlzDetect.

This is the *real* fuzzy logic in the classification path -- distinct from the
skfuzzy resampling controller (which only decides per-class target counts before
training). It is a drop-in Keras layer so the exported `.keras` loads in the
backend with no custom_objects argument (it self-registers on import).

How it works (a Takagi-Sugeno-Kang fuzzy system with Gaussian membership fns):

  1. The CNN feature vector is projected to a small dim D (done outside this
     layer by a Dense projection) so the rule grid stays tractable.
  2. R fuzzy rules. Each rule r has, per input dim d, a Gaussian membership
     function with trainable center c[r,d] and width sigma[r,d]:
        mu[r,d](x) = exp(-0.5 * ((x_d - c[r,d]) / sigma[r,d])^2)
  3. Rule firing strength = product of memberships over the D dims (fuzzy AND):
        f[r](x) = prod_d mu[r,d](x)
  4. Normalize firing strengths (partition of unity), then defuzzify by a
     weighted average of per-rule class consequents w[r, :]:
        out = sum_r (f[r] / sum_k f[k]) * w[r, :]
  5. Softmax -> class probabilities.

Centers, widths and consequents are all learned by backprop, so the membership
functions and rules adapt to the data (this is the ANFIS idea).
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="alzdetect", name="FuzzyLayer")
class FuzzyLayer(layers.Layer):
    def __init__(self, n_rules, n_classes, **kwargs):
        super().__init__(**kwargs)
        self.n_rules = int(n_rules)
        self.n_classes = int(n_classes)

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Gaussian membership functions: one center + width per (rule, input dim).
        self.centers = self.add_weight(
            name="centers", shape=(self.n_rules, d),
            initializer=tf.keras.initializers.RandomNormal(stddev=1.0),
            trainable=True)
        # Parameterize width in log-space to keep it strictly positive.
        self.log_sigma = self.add_weight(
            name="log_sigma", shape=(self.n_rules, d),
            initializer=tf.keras.initializers.Zeros(), trainable=True)
        # Order-0 TSK consequents: a class-logit vector per rule.
        self.consequent = self.add_weight(
            name="consequent", shape=(self.n_rules, self.n_classes),
            initializer="glorot_uniform", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        x = tf.expand_dims(inputs, axis=1)                 # (B, 1, D)
        c = tf.expand_dims(self.centers, axis=0)           # (1, R, D)
        sigma = tf.exp(tf.expand_dims(self.log_sigma, 0))  # (1, R, D), > 0
        # Per-dim Gaussian membership, then fuzzy-AND across dims (in log-space
        # for numerical stability: product of exp = exp of sum).
        log_mu = -0.5 * tf.reduce_sum(tf.square((x - c) / sigma), axis=-1)  # (B, R)
        firing = tf.nn.softmax(log_mu, axis=1)             # normalized strengths
        out = tf.matmul(firing, self.consequent)           # (B, n_classes)
        return tf.nn.softmax(out)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"n_rules": self.n_rules, "n_classes": self.n_classes})
        return cfg
