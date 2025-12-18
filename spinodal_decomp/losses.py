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

            # Crank-Nicolson Scheme (Fully Implicit in Loss)
            # (c - c0)/dt = 0.5 * (RHS(c) + RHS(c0))
            # RHS(u) = M * laplacian(f'(u)) - M * lambda * bi-laplacian(u)
            
            c0_hat = jnp.fft.fft2(c0)
            c_hat = jnp.fft.fft2(c)
            lhs_hat = c_hat - c0_hat

            # M = configs.M
            M = x[1, ...]
            lambda_param = configs.lambda_param
            
            # Calculate nonlinear terms f'(c) = c^3 - c for both time steps
            f_prime_c = c**3 - c
            f_prime_c0 = c0**3 - c0
            
            f_prime_c_hat = jnp.fft.fft2(f_prime_c)
            f_prime_c0_hat = jnp.fft.fft2(f_prime_c0)

            # Spectral derivatives: laplacian -> -K2, bi-laplacian -> K4
            
            # Term 1: M * laplacian(f'(c) + f'(c0)) * 0.5 * dt
            term1_hat = -0.5 * M * K2 * (f_prime_c_hat + f_prime_c0_hat) * dt_unit

            # Term 2: - M * lambda * bi-laplacian(c + c0) * 0.5 * dt
            term2_hat = -0.5 * M * lambda_param * K4 * (c_hat + c0_hat) * dt_unit
            
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
    def ch_loss_real(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:
        
        """
        Compute the Cahn-Hilliard residual in real space.
        """

        def laplacian_fd(u, dx, dy):
            u_ip = jnp.roll(u, -1, axis=-1)
            u_im = jnp.roll(u, 1, axis=-1)
            u_jp = jnp.roll(u, -1, axis=-2)
            u_jm = jnp.roll(u, 1, axis=-2)
            
            d2x = (u_ip + u_im - 2*u) / (dx**2)
            d2y = (u_jp + u_jm - 2*u) / (dy**2)
            return d2x + d2y


        def residual_fn(x, dx, dy, dt):
            pred = model.forward(x)

            c0 = x[0, :, :]
            c = pred[0, :, :]

            dt_phys = dt * configs.Tc
            dx_phys = dx * configs.Lc
            dy_phys = dy * configs.Lc
            M = configs.M
            lambda_param = configs.lambda_param

            def compute_rhs_term(u):
                f_prime = u**3 - u
                lap_u = laplacian_fd(u, dx_phys, dy_phys)
                mu = f_prime - lambda_param * lap_u
                return M * laplacian_fd(mu, dx_phys, dy_phys)

            # Crank-Nicolson Scheme: (c - c0)/dt = 0.5 * (RHS(c) + RHS(c0))
            rhs_c = compute_rhs_term(c)
            rhs_c0 = compute_rhs_term(c0)
            
            # Combine terms: c - c0 - 0.5 * dt * (RHS(c) + RHS(c0))
            rhs_avg = 0.5 * (rhs_c + rhs_c0) * dt_phys
            lhs = c - c0
            residual = lhs - rhs_avg
            return residual
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    
    @classmethod
    def mass_conservation_loss(cls,
                               model: AutoRegressiveModel2d,
                               xs: jnp.ndarray,
                               ys: jnp.ndarray,
                               dx: float,
                               dy: float,
                               **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        mass_pred = jnp.mean(y_pred, axis=(-2, -1))
        mass_true = jnp.mean(ys, axis=(-2, -1))
        loss = jnp.mean(jnp.square(mass_pred - mass_true)) * 100.0  # scale to match other losses
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
        vg_list = VG_FNS
        
        for vg in vg_list:
            (loss, aux_var), grad = vg(
                model, xs, ys=ys, dx=dx, dy=dy, dt=dt, configs=configs
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux_var)

        weights = cls.grad_norm_weights(grads)
        # weights = jnp.array([1.0 for _ in losses])
        # weights = jnp.array([0.0, 1.0]) # try only CH loss
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
        # weights = grad_norms[0] / (grad_norms + eps)
        weights = jnp.sum(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
        
MSE_VG = eqx.filter_value_and_grad(Losses.mse_loss, has_aux=True)
CH_VG  = eqx.filter_value_and_grad(Losses.ch_loss, has_aux=True)
VG_FNS = [CH_VG,]
