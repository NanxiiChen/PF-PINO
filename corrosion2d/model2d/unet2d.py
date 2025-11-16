from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal

from .base_model2d import AutoRegressiveModel2d


class DownBlock2d(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, out_channels, activation, key):
        conv1_key, conv2_key = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, key=conv1_key)
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, key=conv2_key)
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)
        self.activation = activation
        
    def __call__(self, x):
        # First convolution
        x = self.conv1(x)
        x = self.activation(x)
        
        # Second convolution
        x = self.conv2(x)
        x = self.activation(x)
        
        # Store for skip connection
        skip = x
        
        # Pooling for next level
        x = self.pool(x)
        
        return x, skip


class UpBlock2d(eqx.Module):
    upconv: eqx.nn.ConvTranspose2d
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, out_channels, activation, key):
        upconv_key, conv1_key, conv2_key = jax.random.split(key, 3)
        self.upconv = eqx.nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=2, stride=2, key=upconv_key)
        self.conv1 = eqx.nn.Conv2d(
            in_channels, out_channels,  # in_channels because of skip connection concatenation
            kernel_size=3, padding=1, key=conv1_key)
        self.conv2 = eqx.nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1, key=conv2_key)
        self.activation = activation
        
    def __call__(self, x, skip):
        # Upsample
        x = self.upconv(x)
        x = self.activation(x)
        
        # Handle size mismatch due to pooling/upsampling
        if x.shape[-2:] != skip.shape[-2:]:
            # Pad or crop to match skip connection size
            h_diff = skip.shape[-2] - x.shape[-2]
            w_diff = skip.shape[-1] - x.shape[-1]
            
            if h_diff > 0 or w_diff > 0:
                # Need padding
                pad_h = max(0, h_diff)
                pad_w = max(0, w_diff)
                x = jnp.pad(x, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
            
            if h_diff < 0 or w_diff < 0:
                # Need cropping
                end_h = x.shape[-2] + min(0, h_diff)
                end_w = x.shape[-1] + min(0, w_diff)
                x = x[:, :end_h, :end_w]
        
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=0)
        
        # Two convolutions
        x = self.conv1(x)
        x = self.activation(x)
        
        x = self.conv2(x)
        x = self.activation(x)
        
        return x


class UNet2d(AutoRegressiveModel2d):
    # Encoder (downsampling path)
    down_blocks: List[DownBlock2d]
    
    # Bottleneck
    bottleneck_conv1: eqx.nn.Conv2d
    bottleneck_conv2: eqx.nn.Conv2d
    
    # Decoder (upsampling path)
    up_blocks: List[UpBlock2d]
    
    # Final output layer
    output_conv: eqx.nn.Conv2d
    
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, out_channels,
                 width=64, depth=4,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        
        keys = jax.random.split(key, 2 * depth + 3)
        key_idx = 0
        
        self.activation = activation
        
        # Create encoder path
        self.down_blocks = []
        current_channels = in_channels
        
        for i in range(depth):
            out_ch = width * (2 ** i)
            self.down_blocks.append(
                DownBlock2d(current_channels, out_ch, activation, keys[key_idx])
            )
            current_channels = out_ch
            key_idx += 1
        
        # Bottleneck
        bottleneck_channels = width * (2 ** depth)
        self.bottleneck_conv1 = eqx.nn.Conv2d(
            current_channels, bottleneck_channels,
            kernel_size=3, padding=1, key=keys[key_idx])
        key_idx += 1
        
        self.bottleneck_conv2 = eqx.nn.Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, padding=1, key=keys[key_idx])
        key_idx += 1
        
        # Create decoder path
        self.up_blocks = []
        current_channels = bottleneck_channels
        
        for i in range(depth - 1, -1, -1):
            out_ch = width * (2 ** i) if i > 0 else width
            self.up_blocks.append(
                UpBlock2d(current_channels, out_ch, activation, keys[key_idx])
            )
            current_channels = out_ch
            key_idx += 1
        
        # Final output layer
        self.output_conv = eqx.nn.Conv2d(
            current_channels, out_channels,
            kernel_size=1, key=keys[key_idx])
    
    def __call__(self, x):
        # x shape: [channels, height, width]
        
        # Encoder path - store skip connections
        skip_connections = []
        
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skip_connections.append(skip)
        
        # Bottleneck
        x = self.bottleneck_conv1(x)
        x = self.activation(x)
        x = self.bottleneck_conv2(x)
        x = self.activation(x)
        
        # Decoder path - use skip connections in reverse order
        skip_connections.reverse()
        
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[i]
            x = up_block(x, skip)
        
        # Final output layer
        x = self.output_conv(x)
        
        return jax.nn.tanh(x) / 2 + 0.5

