class SwiGLU(tf.keras.layers.Layer):
    def __init__(self, expansion_factor=2, bias=True, dim=-1, **kwargs):
        """
        SwiGLU Activation Layer that automatically adapts to input dimensions
        
        Args:
            expansion_factor: Factor by which to expand the input dimensions (default: 2)
            bias: Whether to use bias in the projection
            dim: The dimension to split on
        """
        super(SwiGLU, self).__init__(**kwargs)
        self.expansion_factor = expansion_factor
        self.bias = bias
        self.dim = dim
        self.dense = None  # Will be initialized on first call
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        # Create a dense layer that projects to expansion_factor * input_dim
        self.dense = tf.keras.layers.Dense(
            self.expansion_factor * input_dim, 
            use_bias=self.bias
        )
        super(SwiGLU, self).build(input_shape)
        
    def call(self, x):
        # Project input to higher dimension
        x = self.dense(x)
        
        # Split the projected tensor into two equal parts
        out, gate = tf.split(x, num_or_size_splits=2, axis=self.dim)
        
        # Apply swish activation to the gate and multiply with output
        gate = tf.nn.silu(gate)  # silu is the same as swish
        x = out * gate
        
        return x
        
    def get_config(self):
        config = super(SwiGLU, self).get_config()
        config.update({
            'expansion_factor': self.expansion_factor,
            'bias': self.bias,
            'dim': self.dim
        })
        return config
 