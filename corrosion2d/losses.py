import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap


from .model2d.base_model2d import AutoRegressiveModel2d


class Losses:

    @classmethod
    @eqx.filter_jit
    def mse_loss(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        loss = jnp.mean((y_pred - ys) ** 2)
        return loss, {}