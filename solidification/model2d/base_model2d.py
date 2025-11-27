import equinox as eqx
import jax.numpy as jnp

class AutoRegressiveModel2d(eqx.Module):
    
    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @eqx.filter_jit
    def forward(self, x, params, **kwargs):
        return self.__call__(x, params, **kwargs)
    
    @eqx.filter_jit
    def auto_reg(self, u0, k, meshes, steps):
        # meshes: [2, Sx, Sy]
        preds = []
        u = u0 # vmap outside the function, so u0 shape is [C, Sx, Sy] without B
        for step in range(steps):

            inputs = jnp.concatenate([u, meshes], 
                                     axis=0)  # [4, Sx, Sy]
            u = self.forward(inputs, k)  # [C, Sx, Sy]
            preds.append(u)
        return jnp.stack(preds, axis=0)  # [T, C, Sx, Sy]


    
    
