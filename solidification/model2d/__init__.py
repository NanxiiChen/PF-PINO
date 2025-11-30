import jax

def get_model2d(model_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create 2D models based on the specified type.
    
    Args:
        model_type (str): Type of the model ('fno', 'unet').
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        **kwargs: Additional model-specific parameters.

    Returns:
        AutoRegressiveModel2d: An instance of the specified model.
    """
    key = kwargs.pop('key', jax.random.PRNGKey(0))

    if model_type.lower() == 'fno':
        from .fno2d import FNO2d
        # FNO model parameters
        modes_x = kwargs.get('modes_x', 16)
        modes_y = kwargs.get('modes_y', 16)
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)

        return FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes_x=modes_x,
            modes_y=modes_y,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
    
    elif model_type.lower() == 'ufno':
        from .fno2d import UFNO2d
        # FNO-UNet model parameters
        modes_x = kwargs.get('modes_x', 16)
        modes_y = kwargs.get('modes_y', 16)
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)

        return UFNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes_x=modes_x,
            modes_y=modes_y,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
    
    elif model_type.lower() == 'unet':
        from .unet2d import UNet2d
        # UNet model parameters
        width = kwargs.get('width', 64)
        depth = kwargs.get('depth', 4)
        activation = kwargs.get('activation', jax.nn.gelu)
        
        return UNet2d(
            in_channels=in_channels,
            out_channels=out_channels,
            width=width,
            depth=depth,
            activation=activation,
            key=key
        )
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}. "
                        f"Supported types: 'fno', 'unet'")
    

__all__ = ['get_model2d']