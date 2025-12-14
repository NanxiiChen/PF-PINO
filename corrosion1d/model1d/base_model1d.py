import equinox as eqx
import jax
import jax.numpy as jnp

class AutoRegressiveModel1d(eqx.Module):

    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @eqx.filter_jit
    def forward(self, x):
        return self.__call__(x)
    
    @eqx.filter_jit
    def auto_reg(self, u0, Lp, meshes, dt, steps):
        # meshes: [S,]
        meshes = meshes[None, :]  # [1, S]
        
        t_channel = jnp.full_like(meshes, dt)
        Lp_channel = jnp.full_like(meshes, Lp)
        
        def scan_fn(u, _):
            inputs = jnp.concatenate([u, Lp_channel, 
                                      meshes, t_channel], 
                                     axis=0)
            u_next = self.forward(inputs)  # [C, S]
            return u_next, u_next

        _, preds = jax.lax.scan(scan_fn, u0, None, length=steps)
        return preds  # [T, C, S]




