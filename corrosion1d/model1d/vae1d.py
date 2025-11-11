from typing import Callable, List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal

from .base_model1d import AutoRegressiveModel1d

class Encoder1d(eqx.Module):
    conv_layers: List[eqx.nn.Conv1d]
    mu_layer: eqx.nn.Linear
    logvar_layer: eqx.nn.Linear
    activation: Callable = jax.nn.relu
    
    def __init__(self, in_channels, latent_dim, 
                 width, depth,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        keys = jax.random.split(key, depth + 2)
        
        # Convolutional layers
        self.conv_layers = []
        channels = in_channels
        for i in range(depth):
            out_channels = width * (2 ** i)
            self.conv_layers.append(
                eqx.nn.Conv1d(
                    channels, out_channels,
                    kernel_size=3, stride=2, padding=1,
                    key=keys[i])
            )
            channels = out_channels
            
        # Latent space layers
        self.mu_layer = eqx.nn.Linear(channels, latent_dim, key=keys[-2])
        self.logvar_layer = eqx.nn.Linear(channels, latent_dim, key=keys[-1])
        self.activation = activation
        
    def __call__(self, x):
        # x shape: [channels, spatial_points]
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        
        # Global average pooling
        x = jnp.mean(x, axis=-1)  # [channels]
        
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

class Decoder1d(eqx.Module):
    linear_layer: eqx.nn.Linear
    deconv_layers: List[eqx.nn.ConvTranspose1d]
    final_conv: eqx.nn.Conv1d
    activation: Callable
    init_size: int
    init_channels: int
    
    def __init__(self, latent_dim, out_channels,
                 width, depth, output_size,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        # Set attributes first
        self.activation = activation
        init_size = output_size // (2 ** depth)
        init_channels = width * (2 ** (depth - 1))
        self.init_size = init_size
        self.init_channels = init_channels
        
        keys = jax.random.split(key, depth + 2)
        
        # Linear layer to project latent to feature maps
        self.linear_layer = eqx.nn.Linear(
            latent_dim, init_channels * init_size, key=keys[0])
        
        # Deconvolutional layers
        self.deconv_layers = []
        channels = init_channels
        for i in range(depth):
            out_ch = channels // 2 if i < depth - 1 else width
            self.deconv_layers.append(
                eqx.nn.ConvTranspose1d(
                    channels, out_ch,
                    kernel_size=4, stride=2, padding=1,
                    key=keys[i + 1])
            )
            channels = out_ch
            
        # Final convolution
        self.final_conv = eqx.nn.Conv1d(
            width, out_channels,
            kernel_size=3, padding=1, key=keys[-1])
        
    def __call__(self, z):
        # z shape: [latent_dim]
        x = self.linear_layer(z)
        x = x.reshape(self.init_channels, self.init_size)
        
        for deconv in self.deconv_layers:
            x = self.activation(deconv(x))
            
        x = self.final_conv(x)
        return x

class VAE1d(AutoRegressiveModel1d):
    encoder: Encoder1d
    decoder: Decoder1d
    latent_dim: int
    
    def __init__(self, in_channels, out_channels,
                 latent_dim=64, width=32, 
                 depth=3, output_size=128,
                 activation=jax.nn.relu,
                 key=jax.random.PRNGKey(0)):
        enc_key, dec_key = jax.random.split(key)
        
        self.encoder = Encoder1d(
            in_channels, latent_dim, 
            width, depth, 
            activation, enc_key)
        self.decoder = Decoder1d(
            latent_dim, out_channels,
            width, depth, output_size,
            activation, dec_key)
        self.latent_dim = latent_dim
        
    def encode(self, x):
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def __call__(self, x, key=jax.random.PRNGKey(0)):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, key)
        reconstructed = self.decode(z)
        
        # Ensure output matches input spatial dimensions
        if reconstructed.shape[-1] != x.shape[-1]:
            # Simple interpolation to match dimensions
            reconstructed = jnp.interp(
                jnp.linspace(0, 1, x.shape[-1]),
                jnp.linspace(0, 1, reconstructed.shape[-1]),
                reconstructed.T
            ).T
            
        return reconstructed, mu, logvar
    
    def vae_loss(self, x, reconstructed, mu, logvar, beta=1.0):
        # Reconstruction loss (MSE)
        recon_loss = jnp.mean((x - reconstructed) ** 2)
        
        # KL divergence loss
        kl_loss = -0.5 * jnp.mean(1 + logvar - mu**2 - jnp.exp(logvar))
        
        return recon_loss + beta * kl_loss
