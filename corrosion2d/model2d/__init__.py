import jax

def get_model2d(model_type, in_channels, out_channels, **kwargs):
    """
    Factory function to create 2D models based on the specified type.
    
    Args:
        model_type (str): Type of the model ('fno', 'fcn', 'unet').
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
        activation = kwargs.get('activation', jax.nn.relu)

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
    
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not implemented for 2D models.")
    

__all__ = ['get_model2d']