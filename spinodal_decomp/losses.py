import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from .model2d.base_model2d import AutoRegressiveModel2d
from .configs.train_debug import Configs


class Spectral2d:
    """
    Spectral Method for 2D derivatives with Periodic BC.
    """
    @staticmethod
    @eqx.filter_jit
    def nabla(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        H, W = u.shape
        u_hat = jnp.fft.fft2(u)
        
        kx = 2 * jnp.pi * jnp.fft.fftfreq(W, d=dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(H, d=dy)
        
        kx = kx[None, :]
        ky = ky[:, None]
        
        dudx = jnp.real(jnp.fft.ifft2(1j * kx * u_hat))
        dudy = jnp.real(jnp.fft.ifft2(1j * ky * u_hat))
        
        return jnp.stack([dudx, dudy], axis=0)
    
    @staticmethod
    @eqx.filter_jit
    def laplacian(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        H, W = u.shape
        u_hat = jnp.fft.fft2(u)
        
        kx = 2 * jnp.pi * jnp.fft.fftfreq(W, d=dx)
        ky = 2 * jnp.pi * jnp.fft.fftfreq(H, d=dy)
        
        kx = kx[None, :]
        ky = ky[:, None]
        
        lap_hat = -(kx**2 + ky**2) * u_hat
        
        return jnp.real(jnp.fft.ifft2(lap_hat))


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
    def mse_loss_weighted(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        # since scale between samples may differ a lot
        # we apply sample-wise  weighting
        y_pred = vmap(model.forward)(xs) # shape (batch, channels, H, W)
        sample_losses = jnp.mean(jnp.square(y_pred - ys), axis=(1,2,3))  # shape (batch,)
        weights = 1.0 / (jnp.sqrt(sample_losses) + 1e-6)
        weights = weights / jnp.sum(weights) * weights.shape[0]  # normalize weights to keep loss scale
        weights = jax.lax.stop_gradient(weights)
        weighted_losses = sample_losses * weights
        loss = jnp.mean(weighted_losses)
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

        def residual_fn(x, dx, dy, dt):
            pred = model.forward(x)

            c0 = x[0, :, :]
            mu0 = x[1, :, :]

            c = pred[0, :, :]
            mu = pred[1, :, :]

            dc_dt = (c - c0) / dt / configs.Tc
            lap_mu = Spectral2d.laplacian(mu, dx, dy) / configs.Lc**2
            M = configs.M
            residual = dc_dt - M * lap_mu
            
            return residual / configs.CH_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}

    @classmethod
    def pot_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:
        """
        Potential equation loss
        """

        def residual_fn(x, dx, dy, dt):
            pred = model.forward(x)

            c0 = x[0, :, :]
            mu0 = x[1, :, :]

            c = pred[0, :, :]
            mu = pred[1, :, :]
            
            f_prime = c**3 - c
            lambda_param = configs.lambda_param
            lap_c = Spectral2d.laplacian(c, dx, dy) / configs.Lc**2
            residual = mu - f_prime + lambda_param * lap_c
            return residual / configs.POT_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    


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
        
        if pde_name == "both":
            vg_list = VG_FNS
        elif pde_name == "ch":
            vg_list = VG_FNS_CH
        elif pde_name == "pot":
            vg_list = VG_FNS_POT
        else:
            raise ValueError(f"Unknown pde_name: {pde_name}")
        
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
        # return total_loss, (losses, weights, aux_vars)
            
    @classmethod
    def grad_norm_weights(cls, grads: list, eps=1e-6):
        def tree_norm(pytree):
            r, _ = ravel_pytree(pytree)   # 一次性把 pytree 展平成 1D 向量
            return jnp.linalg.norm(r)

        grad_norms = jnp.array([tree_norm(g) for g in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = grad_norms[0] / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
        
MSE_VG = eqx.filter_value_and_grad(Losses.mse_loss, has_aux=True)
CH_VG  = eqx.filter_value_and_grad(Losses.ch_loss, has_aux=True)
POT_VG  = eqx.filter_value_and_grad(Losses.pot_loss, has_aux=True)

VG_FNS = [MSE_VG, CH_VG, POT_VG,]
VG_FNS_CH = [MSE_VG, CH_VG,]
VG_FNS_POT = [MSE_VG, POT_VG,]

