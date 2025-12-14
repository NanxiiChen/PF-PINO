import equinox as eqx
import jax
import jax.numpy as jnp

class AutoRegressiveModel2d(eqx.Module):
    
    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @eqx.filter_jit
    def forward(self, x, **kwargs):
        return self.__call__(x, **kwargs)
    
    @eqx.filter_jit
    def auto_reg(self, u0, meshes, steps):
        # meshes: [2, Sx, Sy]
        # u0: [C, Sx, Sy]
        def scan_fn(carry, _):
            u_prev = carry
            inputs = jnp.concatenate([u_prev, meshes], axis=0)
            u_next = self.forward(inputs)
            return u_next, u_next

        _, preds = jax.lax.scan(scan_fn, u0, None, length=steps)
        
        return preds  # [T, C, Sx, Sy]

