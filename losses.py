from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
import optax
import scipy

from model import FNO1d


class FDM1D:
    """
    Finite Difference Derivative Computations in 1D
    """

    @staticmethod
    @eqx.filter_jit
    def first_derivative(u: jnp.ndarray, dx: float) -> jnp.ndarray:
        """
        Compute the first derivative using central difference.

        Args:
            u (jnp.ndarray): Input array of shape [S,], where S is the number of spatial points.
            dx (float): Spatial step size.

        Returns:
            jnp.ndarray: First derivative of u with respect to x, shape [S,].
        """
        dudx = jnp.zeros_like(u)
        dudx = dudx.at[1:-1].set((u[2:] - u[:-2]) / (2 * dx))
        # Forward difference at the first point
        dudx = dudx.at[0].set((u[1] - u[0]) / dx)
        # Backward difference at the last point
        dudx = dudx.at[-1].set((u[-1] - u[-2]) / dx)
        return dudx
    
    @staticmethod
    @eqx.filter_jit
    def second_derivative(u: jnp.ndarray, dx: float) -> jnp.ndarray:
        """
        Compute the second derivative using central difference.

        Args:
            u (jnp.ndarray): Input array of shape [S,], where S is the number of spatial points.
            dx (float): Spatial step size.

        Returns:
            jnp.ndarray: Second derivative of u with respect to x, shape [S,].
        """
        d2udx2 = jnp.zeros_like(u)
        d2udx2 = d2udx2.at[1:-1].set((u[2:] - 2 * u[1:-1] + u[:-2]) / (dx ** 2))
        # Second derivative at the first point using forward difference
        d2udx2 = d2udx2.at[0].set((u[2] - 2 * u[1] + u[0]) / (dx ** 2))
        # Second derivative at the last point using backward difference
        d2udx2 = d2udx2.at[-1].set((u[-1] - 2 * u[-2] + u[-3]) / (dx ** 2))
        return d2udx2


class Losses:

    @classmethod
    @eqx.filter_jit
    def mse_loss(cls, model: FNO1d, 
                 xs: jnp.ndarray, 
                 ys: jnp.ndarray, 
                 **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        return jnp.mean(jnp.square(y_pred - ys)), {}
    
    @classmethod
    def ac_loss(cls, model: FNO1d,
                xs: jnp.ndarray, 
                Lps: jnp.ndarray,
                dx: jnp.ndarray,
                dt: jnp.ndarray,
                configs: object,
                **kwargs) -> jnp.ndarray:
        """
        Allen-Cahn equation loss computation.

        Args:
            model (FNO1d): The neural network model.
            xs (jnp.ndarray): Input array,shape is [B, C+3, S]. B is the batch size, C is number of channels (variables), S is number of spatial points. The extra 3 channels are Lp constant channel, mesh channel, and time channel.
            Lps (jnp.ndarray): Moblity parameter, shape is [B,] vector
            dx (jnp.ndarray): Spatial step size, shape is [1,], scalar
            dt (jnp.ndarray): Time step size, shape is [1,], scalar
            configs (object): Configuration object containing physical parameters.

        Returns:
            jnp.ndarray: Computed Allen-Cahn loss, scalar.
        """

        @eqx.filter_jit
        def residual_fn(x, Lp, dx, dt):
            AC1 = 2 * configs.AA * Lp * configs.Tc
            AC2 = Lp * configs.OMEGA_PHI * configs.Tc
            AC3 = Lp * configs.ALPHA_PHI * configs.Tc / configs.Lc**2
            pred = model.forward(x)

            phi0 = x[0, :]  # phase field variable at current time
            c0 = x[1, :]    # concentration variable at current time

            phi = pred[0, :]  # phase field variable
            c = pred[1, :]    # concentration variable

            h_phi = -2 * phi**3 + 3 * phi**2
            dh_dphi = -6 * phi**2 + 6 * phi
            dg_dphi = 4 * phi**3 - 6 * phi**2 + 2 * phi

            dphi_dt = (phi - phi0) / dt
            lap_phi = FDM1D.second_derivative(phi, dx)
            residual = (
                dphi_dt
                - AC1 * (c - h_phi * (configs.CSE - configs.CLE) - configs.CLE)
                * (configs.CSE - configs.CLE) * dh_dphi
                + AC2 * dg_dphi
                - AC3 * lap_phi
            )
            return residual / configs.AC_PRE_SCALE

        residuals = vmap(residual_fn, in_axes=(0, 0, None, None))(xs, Lps, dx, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    
    @classmethod
    def ch_loss(cls, model: FNO1d,
                xs: jnp.ndarray,
                Lps: jnp.ndarray,
                dx: jnp.ndarray,
                dt: jnp.ndarray,
                configs: object,
                **kwargs) -> jnp.ndarray:
        """
        Cahn-Hilliard equation loss computation.

        Args:
            model (FNO1d): The neural network model.
            xs (jnp.ndarray): Input array,shape is [B, C+3, S]. B is the batch size, C is number of channels (variables), S is number of spatial points. The extra 3 channels are Lp constant channel, mesh channel, and time channel.
            Lps (jnp.ndarray): Moblity parameter, shape is [B,] vector
            dx (jnp.ndarray): Spatial step size, shape is [1,], scalar
            dt (jnp.ndarray): Time step size, shape is [1,], scalar
            configs (object): Configuration object containing physical parameters.

        Returns:
            jnp.ndarray: Computed Cahn-Hilliard loss, scalar.
        """

        @eqx.filter_jit
        def residual_fn(x, Lp, dx, dt):
            CH1 = 2 * configs.AA * configs.MM * configs.Tc / configs.Lc**2
            pred = model.forward(x)

            phi0 = x[0, :]  # phase field variable at current time
            c0 = x[1, :]    # concentration variable at current time

            phi = pred[0, :]  # phase field variable
            c = pred[1, :]    # concentration variable

            nabla_phi = FDM1D.first_derivative(phi, dx)
            dc_dt = (c - c0) / dt
            lap_phi = FDM1D.second_derivative(phi, dx)
            lap_c = FDM1D.second_derivative(c, dx)
            lap_h_phi = 6 * (
                phi * (1 - phi) * lap_phi 
                + (1 - 2 * phi) * nabla_phi**2
            )
            residual = (
                dc_dt
                - CH1 * lap_c
                + CH1 * (configs.CSE - configs.CLE) * lap_h_phi
            )

            return residual

        residuals = vmap(residual_fn, in_axes=(0, 0, None, None))(xs, Lps, dx, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    
    @classmethod
    @eqx.filter_jit
    def pi_loss(cls, model: FNO1d,
                xs: jnp.ndarray,
                ys: jnp.ndarray,
                Lps: jnp.ndarray,
                dx: jnp.ndarray,
                dt: jnp.ndarray,
                configs: object,
                **kwargs) -> jnp.ndarray:
        # total_loss = mse_loss_value + ac_loss_value + ch_loss_value
        
        loss_fns = [cls.mse_loss, cls.ac_loss, cls.ch_loss]
        losses = []
        grads = []
        aux_vars = {}
        for loss_fn in loss_fns:
            (loss, aux_var), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                model,
                xs,
                ys=ys,
                Lps=Lps,
                dx=dx,
                dt=dt,
                configs=configs
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux_var)
            
        weights = cls.grad_norm_weights(grads)
        # weights = weights.at[1].set(0.0)  # set AC loss weight to 0
        # weights = weights.at[2].set(0.0)  # set AC loss weight to 0
        
        total_loss = jnp.sum(jnp.array(weights) * jnp.array(losses))
        return total_loss, (losses, weights, aux_vars)
            
    @classmethod
    def grad_norm_weights(cls, grads: list, eps=1e-6):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)
        grad_norms = jnp.array([tree_norm(grad) for grad in grads])
        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        # weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = grad_norms[0] / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
        

        


    # TODO: total loss fn 
    # TODO: grad_norm_weighting
    # The following code snippet is copied from a legacy repo for reference

    """
    @partial(jit, static_argnums=(0, 4))
    def loss_fn(self, params, batch, eps, pde_name):
        losses, grads, aux_vars = self.compute_losses_and_grads(
            params, batch, eps, pde_name
        )

        weights = self.grad_norm_weights(grads)
        if not self.cfg.IRR:
            weights = weights.at[-1].set(0.0)

        return jnp.sum(weights * losses), (losses, weights, aux_vars)

    @partial(jit, static_argnums=(0,))
    def grad_norm_weights(self, grads: list, eps=1e-6):
        def tree_norm(pytree):
            squared_sum = sum(jnp.sum(x**2) for x in jax.tree_util.tree_leaves(pytree))
            return jnp.sqrt(squared_sum)

        grad_norms = jnp.array([tree_norm(grad) for grad in grads])

        grad_norms = jnp.clip(grad_norms, eps, 1 / eps)
        weights = jnp.mean(grad_norms) / (grad_norms + eps)
        weights = jnp.nan_to_num(weights)
        weights = jnp.clip(weights, eps, 1 / eps)
        return jax.lax.stop_gradient(weights)
    """
    
