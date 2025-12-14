import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from .model2d.base_model2d import AutoRegressiveModel2d
from .configs.train_debug import Configs


class Losses:

    @classmethod
    def mse_loss(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        loss = jnp.mean(jnp.square(y_pred - ys))
        return loss, {}

    @classmethod
    def ch_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:
        
        nx = xs.shape[-1]
        ny = xs.shape[-2]
        dx_phys = dx * configs.Lc
        dy_phys = dy * configs.Lc
        kx = jnp.fft.fftfreq(nx, d=dx_phys) * 2 * jnp.pi
        ky = jnp.fft.fftfreq(ny, d=dy_phys) * 2 * jnp.pi
        KX, KY = jnp.meshgrid(kx, ky, indexing='xy')
        K2 = KX**2 + KY**2
        K4 = K2**2

        def residual_fn(x, dx, dy, dt):
            pred = model.forward(x)

            c0 = x[0, :, :]
            c = pred[0, :, :]

            # use spectral type governing equations
            dt_unit = dt * configs.Tc

            # (c - c0)/dt = M * laplacian(f'(c0)) - M * lambda * bi-laplacian(c)
            # c - c0 = M * laplacian(f'(c0)) * dt - M * lambda * bi-laplacian(c) * dt
            c0_hat = jnp.fft.fft2(c0)
            c_hat = jnp.fft.fft2(c)
            lhs_hat = c_hat - c0_hat

            M = configs.M
            lambda_param = configs.lambda_param
            f_prime = c0**3 - c0 # semi-implicit treatment of f'
            f_prime_hat = jnp.fft.fft2(f_prime)

            # M * laplacian(f'(c0)) -> M * (-K2) * f_prime_hat
            term1_hat = -M * K2 * f_prime_hat * dt_unit

            # - M * lambda * bi-laplacian(c) -> - M * lambda * (K4) * c_hat
            term2_hat = -M * lambda_param * K4 * c_hat * dt_unit
            rhs_hat = term1_hat + term2_hat
            residual_hat = lhs_hat - rhs_hat
            residual = jnp.fft.ifft2(residual_hat).real
            return residual / configs.CH_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {"k2_mean": jnp.mean(K2), 
                      "k4_mean": jnp.mean(K4), 
                      "dx_phys": dx_phys, 
                      "dy_phys": dy_phys, 
                      "dt_phys": dt * configs.Tc}
    
    @classmethod
    @eqx.filter_jit
    def pi_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                ys: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: object,
                pde_name: str = "both",
                **kwargs) -> jnp.ndarray:
        
        losses = []
        grads = []
        aux_vars = {}
        vg_list = VG_FNS
        
        for vg in vg_list:
            (loss, aux_var), grad = vg(
                model, xs, ys=ys, dx=dx, dy=dy, dt=dt, configs=configs
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux_var)

        weights = cls.grad_norm_weights(grads)
        total_loss = jnp.sum(jnp.array(weights) * jnp.array(losses))

        def sum_weighted_grads(weight, grad_tree):
            return jax.tree_map(lambda g: weight * g, grad_tree)
        
        total_grad = jax.tree_map(lambda x: jnp.zeros_like(x), grads[0])
        for i, g in enumerate(grads):
            weighted_g = sum_weighted_grads(weights[i], g)
            total_grad = jax.tree_map(lambda a, b: a + b, total_grad, weighted_g)
            
        return (total_loss, (losses, weights, aux_vars)), total_grad

    @classmethod
    def grad_norm_weights(cls, grads: list, eps=1e-6):
        def tree_norm(pytree):
            r, _ = ravel_pytree(pytree)
            return jnp.linalg.norm(r)

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = grad_norms[0] / (grad_norms + eps)
        # weights = jnp.sum(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
        
MSE_VG = eqx.filter_value_and_grad(Losses.mse_loss, has_aux=True)
CH_VG  = eqx.filter_value_and_grad(Losses.ch_loss, has_aux=True)
VG_FNS = [MSE_VG, CH_VG,]
