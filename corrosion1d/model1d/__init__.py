from .fno1d import FNO1d
from .fcn1d import FCN1d
from .unet1d import UNet1d
from .vae1d import VAE1d
import jax

def get_model1d(model_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create 1D models
    
    Args:
        model_type (str): Model type, supports 'fno', 'fcn', 'unet', 'vae'
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        **kwargs: Model-specific parameters
        
    Returns:
        AutoRegressiveModel1d: Created model instance
    """
    key = kwargs.pop('key', jax.random.PRNGKey(0))
    
    if model_type.lower() == 'fno':
        # FNO model parameters
        modes = kwargs.get('modes', 16)
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)
        
        return FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
        
    elif model_type.lower() == 'fcn':
        # FCN model parameters
        width = kwargs.get('width', 128)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.relu)
        
        return FCN1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
        
    elif model_type.lower() == 'unet':
        # UNet model parameters
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 3)
        activation = kwargs.get('activation', jax.nn.relu)
        
        return UNet1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
        
    elif model_type.lower() == 'vae':
        # VAE model parameters
        latent_dim = kwargs.get('latent_dim', 64)
        width = kwargs.get('width', 32)
        depth = kwargs.get('depth', 3)
        output_size = kwargs.get('output_size', 128)
        activation = kwargs.get('activation', jax.nn.relu)
        
        return VAE1d(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_dim=latent_dim,
            width=width,
            depth=depth,
            output_size=output_size,
            activation=activation,
            key=key
        )
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: 'fno', 'fcn', 'unet', 'vae'")

# Export all model classes and factory function
__all__ = ['FNO1d', 'FCN1d', 'UNet1d', 'VAE1d', 'get_model1d']