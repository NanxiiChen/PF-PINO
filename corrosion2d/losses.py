import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from .model2d.base_model2d import AutoRegressiveModel2d

class FDM2d:
    """
    Finite Difference Method for 2D Corrosion Modeling.
    """
    @staticmethod
    @eqx.filter_jit
    def nabla(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        """
        Compute $\nabla u$ using central difference.
        $\nabla u = (du/dx, du/dy)$
        """
        dudx = jnp.zeros_like(u)
        dudy = jnp.zeros_like(u)

        dudx = dudx.at[:, 1:-1].set((u[:, 2:] - u[:, :-2]) / (2 * dx))
        # dudx = dudx.at[:, 0].set((u[:, 1] - u[:, 0]) / dx)
        # dudx = dudx.at[:, -1].set((u[:, -1] - u[:, -2]) / dx)

        dudy = dudy.at[1:-1, :].set((u[2:, :] - u[:-2, :]) / (2 * dy))
        # dudy = dudy.at[0, :].set((u[1, :] - u[0, :]) / dy)
        # dudy = dudy.at[-1, :].set((u[-1, :] - u[-2, :]) / dy)

        return jnp.stack([dudx, dudy], axis=0)
    
    @staticmethod
    @eqx.filter_jit
    def laplacian(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        """
        Compute $\nabla^2 u$ using central difference.
        $\nabla^2 u = d^2u/dx^2 + d^2u/dy^2$
        """
        d2udx2 = jnp.zeros_like(u)
        d2udy2 = jnp.zeros_like(u)

        d2udx2 = d2udx2.at[:, 1:-1].set((u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2))
        # d2udx2 = d2udx2.at[:, 0].set((2.0*u[:, 0] - 5.0*u[:, 1] + 4.0*u[:, 2] - u[:, 3]) / (dx ** 2))
        # d2udx2 = d2udx2.at[:, -1].set((2.0*u[:, -1] - 5.0*u[:, -2] + 4.0*u[:, -3] - u[:, -4]) / (dx ** 2))

        d2udy2 = d2udy2.at[1:-1, :].set((u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dy ** 2))
        # d2udy2 = d2udy2.at[0, :].set((2.0*u[0, :] - 5.0*u[1, :] + 4.0*u[2, :] - u[3, :]) / (dy ** 2))
        # d2udy2 = d2udy2.at[-1, :].set((2.0*u[-1, :] - 5.0*u[-2, :] + 4.0*u[-3, :] - u[-4, :]) / (dy ** 2))

        return d2udx2 + d2udy2

class Losses:

    @classmethod
    def mse_loss(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        loss = jnp.mean((y_pred - ys) ** 2)
        return loss, {}
    

    @classmethod
    def ac_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: object,
                **kwargs) -> jnp.ndarray:
        """
        Allen-Cahn equation loss for 2D corrosion modeling.
        """

        def residual_fn(x, dx, dy, dt):
            AC1 = 2 * configs.AA * configs.Lp * configs.Tc
            AC2 = configs.Lp * configs.OMEGA_PHI * configs.Tc
            AC3 = configs.Lp * configs.ALPHA_PHI * configs.Tc / configs.Lc**2
            pred = model.forward(x)

            phi0 = x[0, :, :]
            c0 = x[1, :, :]

            phi = pred[0, :, :]
            c = pred[1, :, :]

            h_phi = -2 * phi**3 + 3 * phi**2
            dh_dphi = -6 * phi**2 + 6 * phi
            dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

            dphi_dt = (phi - phi0) / dt
            lap_phi = FDM2d.laplacian(phi, dx, dy)
            residual = (
                dphi_dt
                - AC1 * (c - h_phi * (configs.CSE - configs.CLE) - configs.CLE)
                * (configs.CSE - configs.CLE) * dh_dphi
                + AC2 * dg_dphi
                - AC3 * lap_phi
            )
            return residual / configs.AC_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals[..., 1:-1, 1:-1]))
        return loss, {}
    
    @classmethod
    def ch_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: object,
                **kwargs) -> jnp.ndarray:
        """
        Cahn-Hilliard equation loss for 2D corrosion modeling.
        """

        def residual_fn(x, dx, dy, dt):
            CH1 = 2 * configs.AA * configs.MM * configs.Tc / configs.Lc**2
            pred = model.forward(x)

            phi0 = x[0, :, :]
            c0 = x[1, :, :]

            phi = pred[0, :, :]
            c = pred[1, :, :]

            nabla_phi = FDM2d.nabla(phi, dx, dy)
            dc_dt = (c - c0) / dt
            lap_phi = FDM2d.laplacian(phi, dx, dy)
            lap_c = FDM2d.laplacian(c, dx, dy)
            lap_h_phi = 6 * (
                phi * (1 - phi) * lap_phi 
                + (1 - 2 * phi) * jnp.sum(nabla_phi ** 2, axis=0)
            )
            residual = (
                dc_dt
                - CH1 * lap_c
                + CH1 * (configs.CSE - configs.CLE) * lap_h_phi
            )
            return residual / configs.CH_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals[..., 1:-1, 1:-1]))
        return loss, {}
    
    @classmethod
    def bc_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                dx: float,
                dy: float,
                configs: object,
                **kwargs) -> jnp.ndarray:
        """
        Neumann Boundary Condition loss for 2D corrosion modeling.
        """
        def normal_grad_penalty(u, dx, dy):
            left = (u[:, 1] - u[:, 0])
            right = (u[:, -1] - u[:, -2])
            return (
                jnp.mean(left ** 2) + jnp.mean(right ** 2) 
            )

        def per_sample_bc_loss(x, dx, dy):
            pred = model.forward(x)
            phi = pred[0, :, :]
            c = pred[1, :, :]
            # mu = c - (configs.CSE - configs.CLE) * ( -2 * phi**3 + 3 * phi**2 )
            loss_phi = normal_grad_penalty(phi, dx, dy)
            loss_mu = normal_grad_penalty(c, dx, dy)
            return loss_phi + loss_mu
        
        loss = vmap(per_sample_bc_loss, in_axes=(0, None, None))(xs, dx, dy)
        return jnp.mean(loss), {}
    
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
        for vg in VG_FNS:
            (loss, aux_var), grad = vg(
                model, xs, ys=ys, dx=dx, dy=dy, dt=dt, configs=configs
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux_var)

        weights = cls.grad_norm_weights(grads)
        # Adjust weights based on the PDE being solved
        if pde_name == 'ac':
            weights = jnp.array([weights[0], weights[1], 0.0])
        elif pde_name == 'ch':
            weights = jnp.array([weights[0], 0.0, weights[2]])
        else:
            pass
    
        total_loss = jnp.sum(jnp.array(weights) * jnp.array(losses))
        return total_loss, (losses, weights, aux_vars)
            
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
AC_VG  = eqx.filter_value_and_grad(Losses.ac_loss, has_aux=True)
CH_VG  = eqx.filter_value_and_grad(Losses.ch_loss, has_aux=True)
BC_VG  = eqx.filter_value_and_grad(Losses.bc_loss, has_aux=True)
VG_FNS = [MSE_VG, AC_VG, CH_VG, BC_VG]