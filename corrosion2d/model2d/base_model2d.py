import equinox as eqx
import jax.numpy as jnp

class AutoRegressiveModel2d(eqx.Module):

    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @eqx.filter_jit
    def forward(self, x):
        return self.__call__(x)
    
    @eqx.filter_jit
    def auto_reg(self, u0, meshes, dt, steps):
        # meshes: [Sx, Sy]
        meshes = meshes[None, :]  # [2, Sx, Sy]
        preds = []
        u = u0 # vmap outside the function, so u0 shape is [C, Sx, Sy] without B
        for step in range(steps):
            t_channel = jnp.full(
                (1, meshes.shape[1], meshes.shape[2]),
                dt
            )
            inputs = jnp.concatenate([u, meshes, t_channel], 
                                     axis=0)  # [C+3, Sx, Sy]
            u = self.forward(inputs)  # [C, Sx, Sy]
            preds.append(u)
        return jnp.stack(preds, axis=0)  # [T, C, Sx, Sy]


    
    
