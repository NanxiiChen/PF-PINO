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
    


class FNOBlock2d(eqx.Module):
    spectral_conv: SpectralConv2d
    bypass_conv: eqx.nn.Conv2d
    cond_proj: eqx.nn.Linear
    activation: Callable = jax.nn.relu

    def __init__(self, in_channels, out_channels,
                 modes_x, modes_y, activation, key):
        spec_key, bypass_key, cond_key = jax.random.split(key, 3)
        self.spectral_conv = SpectralConv2d(
            in_channels, out_channels,
            modes_x, modes_y, spec_key)
        self.bypass_conv = eqx.nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, 1), key=bypass_key)
        self.cond_proj = eqx.nn.Linear(
            1, 2 * out_channels, key=cond_key)
        self.activation = activation
        
    def __call__(self, x, params):
        x1 = self.spectral_conv(x)
        x2 = self.bypass_conv(x)
        out = x1 + x2

        style = self.cond_proj(params) # (2 * out_channels,)
        style = style[:, None, None] # 广播到 (2*C, 1, 1) 以便与 (C, H, W) 运算
        scale, shift = jnp.split(style, 2, axis=0) # (C, 1, 1), (C, 1, 1)

        out = out * (1 + scale) + shift

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
        

    def __call__(self, x, params):
        x = self.lifting(x)
        for block in self.fno_blocks:
            x = block(x, params)
        x = self.projection(x)
        return x
                 

# TODO 考虑融合FNO和UNet
# TODO 在频率截断时，引入门控机制，自适应决定选择哪些频率分量

class FNOUNet2d(eqx.Module): # 继承自 eqx.Module 即可
    lifting: eqx.nn.Conv2d
    down_blocks: List[FNOBlock2d]
    mid_block: FNOBlock2d
    up_blocks: List[FNOBlock2d]
    downsamplers: List[eqx.nn.Conv2d] 
    upsamplers: List[eqx.nn.ConvTranspose2d] 
    proj_fusions: List[eqx.nn.Conv2d] # 用于跳跃连接后的特征融合
    projection: eqx.nn.Conv2d
    depth: int
    width: int


    def __init__(self, in_channels, out_channels,
                    modes_x, modes_y, width, depth, 
                    activation=jax.nn.relu,
                    key=jax.random.PRNGKey(0)):
            
            # 1. 计算总共需要的 "顶层" 键的数量
            # N_layers = 2 (lifting, proj) + 4*depth (samplers/fusions) + 1 (mid) + 2*depth (FNO blocks)
            N_keys = 2 + 4 * depth + 2 * depth + 1 
            
            # 2. 分割出所有所需的键
            all_keys = jax.random.split(key, N_keys)
            
            # 3. 逐一分配键
            key_idx = 0
            
            lifting_key = all_keys[key_idx]; key_idx += 1
            self.lifting = eqx.nn.Conv2d(in_channels, width, kernel_size=(1, 1), key=lifting_key)
            
            # --- Down Path ---
            self.down_blocks = []
            self.downsamplers = []
            for i in range(depth):
                block_key = all_keys[key_idx]; key_idx += 1 # 赋给 FNOBlock2d 的父键
                self.down_blocks.append(
                    FNOBlock2d(width, width, modes_x, modes_y, activation, block_key)
                )
                down_key = all_keys[key_idx]; key_idx += 1
                self.downsamplers.append(
                    eqx.nn.Conv2d(width, width, kernel_size=(2, 2), stride=(2, 2), key=down_key)
                )

            # --- Bottleneck ---
            mid_key = all_keys[key_idx]; key_idx += 1
            self.mid_block = FNOBlock2d(width, width, modes_x, modes_y, activation, mid_key)
            
            # --- Up Path ---
            self.up_blocks = []
            self.upsamplers = [] 
            self.proj_fusions = []
            for i in range(depth):
                upsampler_key = all_keys[key_idx]; key_idx += 1
                self.upsamplers.append(
                    eqx.nn.ConvTranspose2d(width, width, kernel_size=(2, 2), stride=(2, 2), key=upsampler_key)
                )
                
                fusion_key = all_keys[key_idx]; key_idx += 1
                self.proj_fusions.append(
                    eqx.nn.Conv2d(2 * width, width, kernel_size=(1, 1), key=fusion_key)
                )
                
                block_key = all_keys[key_idx]; key_idx += 1 # 赋给 FNOBlock2d 的父键
                self.up_blocks.append(
                    FNOBlock2d(width, width, modes_x, modes_y, activation, block_key)
                )

            # --- Projection ---
            proj_key = all_keys[key_idx]; key_idx += 1
            self.projection = eqx.nn.Conv2d(width, out_channels, kernel_size=(1, 1), key=proj_key)
            
            self.depth = depth
            self.width = width
            
            # 验证是否使用了所有的键
            assert key_idx == N_keys, "Key allocation mismatch: Not all keys were used or index calculation was wrong."

    def __call__(self, x, params):
        x = self.lifting(x)
        
        skip_features = []
        
        # --- Down Path ---
        for i in range(self.depth):
            # FNO Block (Preserve skip features *before* downsampling)
            x = self.down_blocks[i](x, params)
            skip_features.append(x)
            # Downsampling
            x = self.downsamplers[i](x)
            
        # --- Bottleneck ---
        x = self.mid_block(x, params)
        
        # --- Up Path ---
        for i in range(self.depth - 1, -1, -1):
            # 1. Upsample using ConvTranspose2d
            x = self.upsamplers[i](x) 
            
            # 2. Skip Connection & Fusion
            # 确保空间维度匹配，ConvTranspose可能会导致边界问题，
            # 这里我们假设它与 skip_features[i] 的尺寸精确匹配或需要裁剪/填充。
            # 如果尺寸不匹配，需要 JAX 的切片或填充操作。
            # 假设 ConvTranspose2d(k=2, s=2) 完美恢复尺寸
            
            skip = skip_features[i]
            
            # 如果上采样的尺寸略大，进行裁剪（常见于 U-Net 实现）
            if x.shape[-2] > skip.shape[-2] or x.shape[-1] > skip.shape[-1]:
                # 假设只多 1 个像素
                x = x[:, :skip.shape[-2], :skip.shape[-1]] 
            elif x.shape[-2] < skip.shape[-2] or x.shape[-1] < skip.shape[-1]:
                 # 如果上采样的尺寸略小，通常是实现错误或需要填充
                 raise ValueError("Upsampled feature map is smaller than skip feature map.")
                
            x = jnp.concatenate([x, skip], axis=0) # (2*W, H, W)
            
            # Feature Fusion (1x1 Conv)
            x = self.proj_fusions[i](x) 
            
            # 3. FNO Block
            x = self.up_blocks[i](x, params)

        # --- Projection ---
        x = self.projection(x)
        return x