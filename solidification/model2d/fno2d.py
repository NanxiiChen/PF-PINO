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
    

class MixedConv2d(eqx.Module):
    """
    混合尺度卷积模块：并行执行 1x1 和 3x3 卷积。
    - 1x1 卷积：类似于全连接层，负责通道间的信息融合，有助于捕捉极高频的点特征。
    - 3x3 卷积：负责捕捉局部空间纹理和边缘信息。
    """
    conv_1x1: eqx.nn.Conv2d
    conv_3x3: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, activation, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv_1x1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), key=k1)
        self.conv_3x3 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, key=k2)

    def __call__(self, x):
        return self.conv_1x1(x) + self.conv_3x3(x)

class UNetBypassBlock(eqx.Module):
    enc: eqx.nn.Conv2d
    bottleneck: eqx.nn.Conv2d
    dec: eqx.nn.Conv2d
    activation: Callable = jax.nn.relu

    def __init__(self, in_channels, out_channels, activation, key):
        k1, k2, k3 = jax.random.split(key, 3)
        mid = out_channels // 2
        
        # 下采样卷积 (Stride=2)
        self.enc = eqx.nn.Conv2d(in_channels, mid, kernel_size=3, stride=2, padding=1, key=k1)
        
        # 底部卷积
        self.bottleneck = eqx.nn.Conv2d(mid, mid, kernel_size=3, padding=1, key=k2)
        
        # 上采样后的融合卷积 (输入是 bottleneck输出 + 原始输入x)
        # 注意：这里我们利用 x 作为 skip connection
        self.dec = eqx.nn.Conv2d(mid + in_channels, out_channels, kernel_size=3, padding=1, key=k3)
        self.activation = activation

    def __call__(self, x):
        # x: (C, H, W)
        
        # 1. Downsample
        x_down = self.activation(self.enc(x)) # (mid, H/2, W/2)
        
        # 2. Bottleneck
        x_bot = self.activation(self.bottleneck(x_down)) # (mid, H/2, W/2)
        
        # 3. Upsample
        x_up = jax.image.resize(x_bot, (x_bot.shape[0], x.shape[1], x.shape[2]), method='bilinear') # (mid, H, W)
        
        # 4. Skip Connection (Concatenate with original input)
        x_cat = jnp.concatenate([x_up, x], axis=0) # (mid + C, H, W)
        
        # 5. Final Conv
        return self.dec(x_cat)


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
        x = (x + jnp.flip(x, axis=-2)) / 2.0
        x = (x + jnp.flip(x, axis=-1)) / 2.0
        return x
                 