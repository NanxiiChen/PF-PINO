from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal


class SpectralConv1d(eqx.Module):
    real_weights: jnp.ndarray
    imag_weights: jnp.ndarray
    in_channels: int
    out_channels: int
    modes: int
    
    def __init__(self, in_channels, out_channels, modes, key):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        real_key, imag_key = jax.random.split(key)
        self.real_weights = glorot_normal(
            )(real_key, (in_channels, out_channels, modes))
        self.imag_weights = glorot_normal(
            )(imag_key, (in_channels, out_channels, modes))
    
    def __call__(self, x):
        channels, spatial_points = x.shape # batched along `samples` dimension
        x_rft = jnp.fft.rfft(x, axis=-1) # (in_channels, spatial_points//2 + 1)
        assert x_rft.shape[1] >= self.modes, \
            "modes must be less than or equal to half of spatial points"
        x_rft_cut = x_rft[:, :self.modes] # (in_channels, modes)
        weights = self.real_weights + 1j * self.imag_weights # (in_channels, out_channels, modes)
        out_rft_cut = jnp.einsum("im,iom->om", x_rft_cut, weights) # (out_channels, modes)
        out_rft = jnp.zeros(
            (self.out_channels, spatial_points//2 + 1), 
            dtype=x_rft.dtype)
        out_rft = out_rft.at[:, :self.modes].set(out_rft_cut)
        out = jnp.fft.irfft(out_rft, n=spatial_points, axis=-1) # (out_channels, spatial_points)
        return out
    
class FNOBlock1d(eqx.Module):
    spectral_conv: SpectralConv1d
    bypass_conv: eqx.nn.Conv1d
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, out_channels,
                 modes, activation, key):
        spec_key, bypass_key = jax.random.split(key)
        self.spectral_conv = SpectralConv1d(
            in_channels, out_channels, 
            modes, spec_key)
        self.bypass_conv = eqx.nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, key=bypass_key)
        self.activation = activation
        
    def __call__(self, x):
        x1 = self.spectral_conv(x)
        x2 = self.bypass_conv(x)
        return self.activation(x1 + x2)
        
        

class FNO1d(eqx.Module):
    lifting: eqx.nn.Conv1d
    fno_blocks: List[FNOBlock1d]
    projection: eqx.nn.Conv1d
    
    def __init__(self, in_channels, out_channels,
                 modes, width, depth, 
                 activation=jax.nn.gelu,
                 key=jax.random.PRNGKey(0)):
        lifting_key, proj_key, *block_keys = jax.random.split(key, depth + 2)
        self.lifting = eqx.nn.Conv1d(
            in_channels, width,
            kernel_size=1, key=lifting_key)
        self.fno_blocks = [
            FNOBlock1d(
                width, width,
                modes, activation, block_keys[i])
            for i in range(depth)
        ]
        self.projection = eqx.nn.Conv1d(
            width, out_channels,
            kernel_size=1, key=proj_key)
        
    def __call__(self, x):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x)
        x = self.projection(x)
        return x
    
    # def auto_reg(self, x, steps):
    #     preds = []
    #     for _ in range(steps):
    #         x = self.__call__(x)
    #         preds.append(x)
    #     return jnp.stack(preds, axis=0)

    @eqx.filter_jit
    def forward(self, x):
        return self.__call__(x)

    
    @eqx.filter_jit
    def auto_reg(self, u0, Lp, meshes, dt, steps):
        # meshes: [S,]
        meshes = meshes[None, :]  # [1, S]
        preds = []
        u = u0 # vmap outside the function, so u0 shape is [C, S] without B
        for step in range(steps):
            tic = step * dt
            t_channel = jnp.full_like(meshes, tic)
            Lp_channel = jnp.full_like(meshes, Lp)
            inputs = jnp.concatenate([u, Lp_channel, 
                                      meshes, t_channel], 
                                     axis=0)  # [C+2, S]
            u = self.forward(inputs)  # [C, S]
            preds.append(u)
        return jnp.stack(preds, axis=0)  # [T, C, S]
            
            