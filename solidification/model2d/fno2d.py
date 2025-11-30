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
    Mixed Scale Convolution Module: performs 1x1 and 3x3 convolutions in parallel.
    - 1x1 Convolution: acts like a fully connected layer, facilitating inter-channel information fusion, helping to capture very high-frequency point features.
    - 3x3 Convolution: captures local spatial textures and edge information.
    """
    conv_1x1: eqx.nn.Conv2d
    conv_3x3: eqx.nn.Conv2d

    def __init__(self, in_channels, out_channels, activation, key):
        k1, k2 = jax.random.split(key, 2)
        self.conv_1x1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), key=k1)
        self.conv_3x3 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1, key=k2)

    def __call__(self, x):
        return self.conv_1x1(x) + self.conv_3x3(x)


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
        phi, T = jnp.split(x, 2, axis=0)
        phi = jnp.tanh(phi)
        T = T / 3 - 0.3
        x = jnp.concatenate([phi, T], axis=0)
        return x
 

