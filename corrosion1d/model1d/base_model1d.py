import equinox as eqx
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
        preds = []
        u = u0 # vmap outside the function, so u0 shape is [C, S] without B
        for step in range(steps):
            t_channel = jnp.full_like(meshes, dt)
            Lp_channel = jnp.full_like(meshes, Lp)
            inputs = jnp.concatenate([u, Lp_channel, 
                                      meshes, t_channel], 
                                     axis=0)  # [C+2, S]
            u = self.forward(inputs)  # [C, S]
            preds.append(u)
        return jnp.stack(preds, axis=0)  # [T, C, S]
            

    
    
