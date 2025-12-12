from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal

from .base_model2d import AutoRegressiveModel2d


class SpectralConv2d(eqx.Module):
    real_weights_pos: jnp.ndarray
    imag_weights_pos: jnp.ndarray
    real_weights_neg: jnp.ndarray
    imag_weights_neg: jnp.ndarray
    in_channels: int
    out_channels: int
    modes_x: int
    modes_y: int

    def __init__(self, in_channels, out_channels, modes_x, modes_y, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.real_weights_pos = glorot_normal(
            )(k1, (in_channels, out_channels, modes_x, modes_y))
        self.imag_weights_pos = glorot_normal(
            )(k2, (in_channels, out_channels, modes_x, modes_y))
        self.real_weights_neg = glorot_normal(
            )(k3, (in_channels, out_channels, modes_x, modes_y))
        self.imag_weights_neg = glorot_normal(
            )(k4, (in_channels, out_channels, modes_x, modes_y))
        
    def __call__(self, x):
        channels, size_x, size_y = x.shape # batched along `samples` dimension
        x_rft = jnp.fft.rfftn(x, axes=(-2, -1)) # (in_channels, size_x, size_y//2 + 1)
        assert x_rft.shape[1] >= self.modes_x and x_rft.shape[2] >= self.modes_y, \
            "modes must be less than or equal to half of spatial points"
        x_rft_cut_pos = x_rft[:, :self.modes_x, :self.modes_y] # (in_channels, modes_x, modes_y)
        x_rft_cut_neg = x_rft[:, -self.modes_x:, :self.modes_y] # (in_channels, modes_x, modes_y)
        weights_pos = self.real_weights_pos + 1j * self.imag_weights_pos # (in_channels, out_channels, modes_x, modes_y)
        weights_neg = self.real_weights_neg + 1j * self.imag_weights_neg # (in_channels, out_channels, modes_x, modes_y)
        out_rft_cut_pos = jnp.einsum("imn,iomn->omn", x_rft_cut_pos, weights_pos) # (out_channels, modes_x, modes_y)
        out_rft_cut_neg = jnp.einsum("imn,iomn->omn", x_rft_cut_neg, weights_neg) # (out_channels, modes_x, modes_y)
        out_rft = jnp.zeros(
            (self.out_channels, size_x, size_y//2 + 1),
            dtype=x_rft.dtype)
        out_rft = out_rft.at[:, :self.modes_x, :self.modes_y].set(out_rft_cut_pos)
        out_rft = out_rft.at[:, -self.modes_x:, :self.modes_y].set(out_rft_cut_neg)
        return jnp.fft.irfftn(out_rft, s=(size_x, size_y), axes=(-2, -1))
    

def circular_pad(x, padding):
    """
    对输入 x 进行循环填充 (Periodic Boundary Condition)。
    x shape: (channels, h, w)
    """
    if padding == 0:
        return x

    return jnp.pad(x, ((0, 0), (padding, padding), (padding, padding)), mode='wrap')


class MixedConv2d(eqx.Module):
    conv_1x1: eqx.nn.Conv2d
    conv_3x3: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, activation, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv_1x1 = eqx.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1), key=k1
        )
        self.conv_3x3 = eqx.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(3, 3), padding=0, key=k2
        )

    def __call__(self, x):
        out1 = self.conv_1x1(x)
        
        # 3x3 卷积前先做循环填充 (padding=1)
        x_pad = circular_pad(x, 1)
        out2 = self.conv_3x3(x_pad)
        
        return out1 + out2


class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: MixedConv2d
    activation: Callable = jax.nn.relu

    def __init__(self, in_channels, out_channels,
                 modes_x, modes_y, activation, key):
        spec_key, bypass_key = jax.random.split(key, 2)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels,
            modes_x, modes_y, spec_key)
        self.bypass_conv = MixedConv2d(
            in_channels, out_channels, 
            activation, bypass_key)
        self.activation = activation
        
    def __call__(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.bypass_conv(x)
        out = x1 + x2
        return self.activation(out)
    


# class LightweightUNet(eqx.Module):
#     """
#     轻量级U-Net，专门捕获高频细节
#     """
#     down_conv1: eqx.nn.Conv2d
#     down_conv2: eqx.nn.Conv2d
#     bottleneck: eqx.nn.Conv2d
#     up_conv1: eqx.nn.Conv2d
#     up_conv2: eqx.nn.Conv2d
    
#     def __init__(self, in_channels, hidden_dim=16, key=None):
#         """
#         hidden_dim: 保持很小(8-16)，只捕获必要的高频信息
#         """
#         k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        
#         # Encoder: 逐步下采样，增加感受野
#         self.down_conv1 = eqx.nn.Conv2d(
#             in_channels, hidden_dim, 
#             kernel_size=(3, 3), padding=0, key=k1
#         )
#         self.down_conv2 = eqx.nn.Conv2d(
#             hidden_dim, hidden_dim * 2,
#             kernel_size=(3, 3), padding=0, key=k2
#         )
        
#         # Bottleneck
#         self.bottleneck = eqx.nn.Conv2d(
#             hidden_dim * 2, hidden_dim * 2,
#             kernel_size=(3, 3), padding=0, key=k3
#         )
        
#         # Decoder: 对称上采样
#         self.up_conv1 = eqx.nn.Conv2d(
#             hidden_dim * 4, hidden_dim,  # *4 因为有skip connection
#             kernel_size=(3, 3), padding=0, key=k4
#         )
#         self.up_conv2 = eqx.nn.Conv2d(
#             hidden_dim * 2, in_channels,  # 输出和输入同通道
#             kernel_size=(3, 3), padding=0, key=k5
#         )
    
#     def __call__(self, x):
#         # Encoder
#         x1 = circular_pad(x, 1)
#         x1 = jax.nn.relu(self.down_conv1(x1))  # (hidden_dim, H, W)
        
#         x2 = circular_pad(x1, 1)
#         x2 = jax.nn.relu(self.down_conv2(x2))  # (hidden_dim*2, H, W)
        
#         # Bottleneck
#         x_bottle = circular_pad(x2, 1)
#         x_bottle = jax.nn.relu(self.bottleneck(x_bottle))  # (hidden_dim*2, H, W)
        
#         # Decoder with skip connections
#         x_up1 = jnp.concatenate([x_bottle, x2], axis=0)  # Skip from encoder
#         x_up1 = circular_pad(x_up1, 1)
#         x_up1 = jax.nn.relu(self.up_conv1(x_up1))  # (hidden_dim, H, W)
        
#         x_up2 = jnp.concatenate([x_up1, x1], axis=0)  # Skip from encoder
#         x_up2 = circular_pad(x_up2, 1)
#         x_out = self.up_conv2(x_up2)  # (in_channels, H, W)
        
#         return x_out



class FNO2d(AutoRegressiveModel2d):
    lifting: eqx.nn.Conv2d
    fno_blocks: List[FNOBlock2d]
    projection: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels,
                 modes_x, modes_y,
                 width, depth,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        
        lifting_key, proj_key, *block_keys = jax.random.split(key, depth + 2)
        self.lifting = eqx.nn.Conv2d(
            in_channels, width,
            kernel_size=(1, 1), key=lifting_key)
        self.fno_blocks = []
        for i in range(depth):
            self.fno_blocks.append(
                FNOBlock2d(
                    width, width,
                    modes_x, modes_y,
                    activation, block_keys[i])
                )
        self.projection = eqx.nn.Conv2d(
            width, out_channels,
            kernel_size=(1, 1), key=proj_key)
        

    def __call__(self, x):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.projection(x)
        return x 