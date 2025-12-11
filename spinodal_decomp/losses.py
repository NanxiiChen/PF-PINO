import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from .model2d.base_model2d import AutoRegressiveModel2d
from .configs.train_debug import Configs


# class FDM2d:
#     """
#     Finite Difference Method for 2D Corrosion Modeling.
#     """
#     @staticmethod
#     @eqx.filter_jit
#     def nabla(
#         u: jnp.ndarray,
#         dx: float,
#         dy: float
#     ) -> jnp.ndarray:
#         r"""
#         Compute $\nabla u$ using central difference.
#         $\nabla u = (du/dx, du/dy)$
#         """
#         dudx = jnp.zeros_like(u)
#         dudy = jnp.zeros_like(u)

#         # 首先，x和y坐标对应数组的列和行
#         # 其次，y的正方向本来是向上的，但是为了和数组的索引向下对应，我们的meshy实际上也是从上到下增大的
#         dudx = dudx.at[:, 1:-1].set((u[:, 2:] - u[:, :-2]) / (2 * dx))
#         dudy = dudy.at[1:-1, :].set((u[2:, :] - u[:-2, :]) / (2 * dy))

#         dudx = dudx.at[:, 0].set((-3*u[:, 0] + 4*u[:, 1] - u[:, 2]) / (2*dx))
#         dudx = dudx.at[:, -1].set((3*u[:, -1] - 4*u[:, -2] + u[:, -3]) / (2*dx))

#         dudy = dudy.at[0, :].set((-3*u[0, :] + 4*u[1, :] - u[2, :]) / (2*dy))
#         dudy = dudy.at[-1, :].set((3*u[-1, :] - 4*u[-2, :] + u[-3, :]) / (2*dy))


#         return jnp.stack([dudx, dudy], axis=0)
    
    
#     @staticmethod
#     @eqx.filter_jit
#     def laplacian(
#         u: jnp.ndarray,
#         dx: float,
#         dy: float
#     ) -> jnp.ndarray:
#         """
#         Compute $\nabla^2 u$ using central difference.
#         $\nabla^2 u = d^2u/dx^2 + d^2u/dy^2$
#         """
#         d2udx2 = jnp.zeros_like(u)
#         d2udy2 = jnp.zeros_like(u)

#         d2udx2 = d2udx2.at[:, 1:-1].set((u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2))
#         d2udx2 = d2udx2.at[:, 0].set((2.0*u[:, 0] - 5.0*u[:, 1] + 4.0*u[:, 2] - u[:, 3]) / (dx ** 2))
#         d2udx2 = d2udx2.at[:, -1].set((2.0*u[:, -1] - 5.0*u[:, -2] + 4.0*u[:, -3] - u[:, -4]) / (dx ** 2))

#         d2udy2 = d2udy2.at[1:-1, :].set((u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dy ** 2))
#         d2udy2 = d2udy2.at[0, :].set((2.0*u[0, :] - 5.0*u[1, :] + 4.0*u[2, :] - u[3, :]) / (dy ** 2))
#         d2udy2 = d2udy2.at[-1, :].set((2.0*u[-1, :] - 5.0*u[-2, :] + 4.0*u[-3, :] - u[-4, :]) / (dy ** 2))

#         return d2udx2 + d2udy2

class FDM2d:
    """
    Finite Difference Method for 2D Corrosion Modeling with Periodic BC.
    """
    @staticmethod
    @eqx.filter_jit
    def nabla(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        r"""
        Compute $\nabla u$ using central difference with periodic BC.
        $\nabla u = (du/dx, du/dy)$
        """
        dudx = jnp.zeros_like(u)
        dudy = jnp.zeros_like(u)

        # 内部点：中心差分
        dudx = dudx.at[:, 1:-1].set((u[:, 2:] - u[:, :-2]) / (2 * dx))
        dudy = dudy.at[1:-1, :].set((u[2:, :] - u[:-2, :]) / (2 * dy))

        # 周期边界条件
        # x方向：左边界使用右边界的点，右边界使用左边界的点
        dudx = dudx.at[:, 0].set((u[:, 1] - u[:, -1]) / (2 * dx))
        dudx = dudx.at[:, -1].set((u[:, 0] - u[:, -2]) / (2 * dx))

        # y方向：上边界使用下边界的点，下边界使用上边界的点
        dudy = dudy.at[0, :].set((u[1, :] - u[-1, :]) / (2 * dy))
        dudy = dudy.at[-1, :].set((u[0, :] - u[-2, :]) / (2 * dy))

        return jnp.stack([dudx, dudy], axis=0)
    
    
    @staticmethod
    @eqx.filter_jit
    def laplacian(
        u: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        """
        Compute $\nabla^2 u$ using central difference with periodic BC.
        $\nabla^2 u = d^2u/dx^2 + d^2u/dy^2$
        """
        d2udx2 = jnp.zeros_like(u)
        d2udy2 = jnp.zeros_like(u)

        # 内部点
        d2udx2 = d2udx2.at[:, 1:-1].set((u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]) / (dx ** 2))
        d2udy2 = d2udy2.at[1:-1, :].set((u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dy ** 2))

        # 周期边界条件
        # x方向
        d2udx2 = d2udx2.at[:, 0].set((u[:, 1] - 2.0 * u[:, 0] + u[:, -1]) / (dx ** 2))
        d2udx2 = d2udx2.at[:, -1].set((u[:, 0] - 2.0 * u[:, -1] + u[:, -2]) / (dx ** 2))

        # y方向
        d2udy2 = d2udy2.at[0, :].set((u[1, :] - 2.0 * u[0, :] + u[-1, :]) / (dy ** 2))
        d2udy2 = d2udy2.at[-1, :].set((u[0, :] - 2.0 * u[-1, :] + u[-2, :]) / (dy ** 2))

        return d2udx2 + d2udy2


class Losses:
    # TODO: consider adding sample-wise weight for hard samples
    # e.g.: hard-mining loss
    # we can select the top-k hardest samples in a batch to compute the loss
    # say top 20% samples with highest mse loss or relative l2 error
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
            lap_mu = FDM2d.laplacian(mu, dx, dy) / configs.Lc**2
            M = configs.M
            residual = dc_dt - M * lap_mu

            return residual / configs.CH_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, None, None, None))(xs, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}

    @classmethod
    def bc_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                **kwargs) -> jnp.ndarray:
        """
        Periodic Boundary Condition Loss
        """

        def residual_fn(x):
            pred = model.forward(x)

            c = pred[0, :, :]
            mu = pred[1, :, :]

            # Left-Right BC
            bc_lr_c = c[:, 0] - c[:, -1]
            bc_lr_mu = mu[:, 0] - mu[:, -1]

            # Top-Bottom BC
            bc_tb_c = c[0, :] - c[-1, :]
            bc_tb_mu = mu[0, :] - mu[-1, :]

            residual = jnp.concatenate([bc_lr_c, bc_lr_mu, bc_tb_c, bc_tb_mu], axis=0)
            return residual
        
        residuals = vmap(residual_fn)(xs)
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
            
            c_mid = 0.5 * (c + c0)
            f_prime = c_mid**3 - c_mid
            lambda_param = configs.lambda_param
            lap_c = FDM2d.laplacian(c_mid, dx, dy) / configs.Lc**2
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
# BC_VG  = eqx.filter_value_and_grad(Losses.bc_loss, has_aux=True)

VG_FNS = [MSE_VG, CH_VG, POT_VG,]
VG_FNS_CH = [MSE_VG, CH_VG,]
VG_FNS_POT = [MSE_VG, POT_VG,]

