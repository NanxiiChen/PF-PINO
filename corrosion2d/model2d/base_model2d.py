import equinox as eqx
import jax
import jax.numpy as jnp

class AutoRegressiveModel2d(eqx.Module):
    
    def __call__(self, x):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    @eqx.filter_jit
    def forward(self, x, **kwargs):
        # cle = kwargs.get('cle', 5100 / 1.43e5)
        # cse = kwargs.get('cse', 1.0)
        # y = jnp.tanh(self.__call__(x)) / 2 + 0.5
        # phi = y[0:1, :, :]
        # cl = y[1:2, :, :] * (1 - cse + cle)
        # c = (cse - cle) * (-2 * phi**3 + 3 * phi**2) + cl
        # return jnp.concatenate([phi, c], axis=0)
        return self.__call__(x)
    
    @eqx.filter_jit
    def auto_reg(self, u0, meshes, dt, steps):
        # meshes: [2, Sx, Sy]
        t_channel = jnp.full(
            (1, meshes.shape[1], meshes.shape[2]),
            dt
        )
        
        def scan_fn(u, _):
            inputs = jnp.concatenate([u, meshes, t_channel], 
                                     axis=0)  # [C+3, Sx, Sy]
            u_next = self.forward(inputs)  # [C, Sx, Sy]
            return u_next, u_next

        _, preds = jax.lax.scan(scan_fn, u0, None, length=steps)
        return preds  # [T, C, Sx, Sy]




