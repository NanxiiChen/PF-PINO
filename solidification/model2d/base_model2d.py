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
    def auto_reg(self, u0, k, meshes, steps):
        # meshes: [2, Sx, Sy]
        k_channel = jnp.full(
            (1, meshes.shape[1], meshes.shape[2]),
            k
        )
        
        def scan_fn(u, _):
            inputs = jnp.concatenate([u, k_channel, meshes], 
                                     axis=0)  # [C+3, Sx, Sy]
            u_next = self.forward(inputs)  # [C, Sx, Sy]
            return u_next, u_next

        _, preds = jax.lax.scan(scan_fn, u0, None, length=steps)
        return preds  # [T, C, Sx, Sy]





