import jax

def get_model1d(model_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create 1D models
    
    Args:
        model_type (str): Model type, supports 'fno', 'fcn', 'unet'
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        **kwargs: Model-specific parameters
        
    Returns:
        AutoRegressiveModel1d: Created model instance
    """
    key = kwargs.pop('key', jax.random.PRNGKey(0))
    
    if model_type.lower() == 'fno':
        from .fno1d import FNO1d
        # FNO model parameters
        modes = kwargs.get('modes', 16)
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)
        inception = kwargs.get('inception', True)
        
        return FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            depth=depth,
            activation=activation,
            key=key,
            inception=inception
        )
        
    elif model_type.lower() == 'fcn':
        from .fcn1d import FCN1d
        # FCN model parameters
        width = kwargs.get('width', 128)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)
        
        return FCN1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
        
    elif model_type.lower() == 'unet':
        from .unet1d import UNet1d
        # UNet model parameters
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 3)
        activation = kwargs.get('activation', jax.nn.gelu)
        
        return UNet1d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
        
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: 'fno', 'fcn', 'unet'")

# Export all model classes and factory function
__all__ = ['get_model1d']