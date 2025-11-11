from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal

from .base_model1d import AutoRegressiveModel1d

class FCN1d(AutoRegressiveModel1d):
    convs: List[eqx.nn.Conv1d]
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, out_channels,
                 width, depth,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, depth + 1)
        
        # Create layer dimensions (using 1x1 convolutions as fully connected layers)
        layer_dims = [in_channels] + [width] * depth + [out_channels]
        
        # Create 1x1 convolution layers (equivalent to FCN for 1D)
        self.convs = []
        for i in range(len(layer_dims) - 1):
            self.convs.append(
                eqx.nn.Conv1d(
                    layer_dims[i], layer_dims[i + 1],
                    kernel_size=1, key=keys[i])
            )
        
        self.activation = activation
        
    def __call__(self, x):
        # x shape: [channels, spatial_points]
        
        # Apply 1x1 convolutions (equivalent to pointwise FCN)
        for i, conv in enumerate(self.convs):
            x = conv(x)
            if i < len(self.convs) - 1:  # Don't apply activation to output layer
                x = self.activation(x)
        
        return x