import equinox as eqx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.flatten_util import ravel_pytree
from .model2d.base_model2d import AutoRegressiveModel2d
from .configs.train_debug import Configs

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
        r"""
        Compute $\nabla u$ using central difference.
        $\nabla u = (du/dx, du/dy)$
        """
        dudx = jnp.zeros_like(u)
        dudy = jnp.zeros_like(u)

        # 首先，x和y坐标对应数组的列和行
        # 其次，y的正方向本来是向上的，但是为了和数组的索引向下对应，我们的meshy实际上也是从上到下增大的
        dudx = dudx.at[:, 1:-1].set((u[:, 2:] - u[:, :-2]) / (2 * dx))
        dudy = dudy.at[1:-1, :].set((u[2:, :] - u[:-2, :]) / (2 * dy))

        dudx = dudx.at[:, 0].set((-3*u[:, 0] + 4*u[:, 1] - u[:, 2]) / (2*dx))
        dudx = dudx.at[:, -1].set((3*u[:, -1] - 4*u[:, -2] + u[:, -3]) / (2*dx))

        dudy = dudy.at[0, :].set((-3*u[0, :] + 4*u[1, :] - u[2, :]) / (2*dy))
        dudy = dudy.at[-1, :].set((3*u[-1, :] - 4*u[-2, :] + u[-3, :]) / (2*dy))


        return jnp.stack([dudx, dudy], axis=0)
    
    @staticmethod
    @eqx.filter_jit
    def divergence(
        vec_field: jnp.ndarray,
        dx: float,
        dy: float
    ) -> jnp.ndarray:
        r"""
        Compute $\nabla \cdot \mathbf{F}$ using central difference.
        $\nabla \cdot \mathbf{F} = dF_x/dx + dF_y/dy$
        """
        Fx = vec_field[0, :, :]
        Fy = vec_field[1, :, :]

        dFxdx = jnp.zeros_like(Fx)
        dFydy = jnp.zeros_like(Fy)

        dFxdx = dFxdx.at[:, 1:-1].set((Fx[:, 2:] - Fx[:, :-2]) / (2 * dx))
        dFxdx = dFxdx.at[:, 0].set((-3*Fx[:, 0] + 4*Fx[:, 1] - Fx[:, 2]) / (2*dx))
        dFxdx = dFxdx.at[:, -1].set((3*Fx[:, -1] - 4*Fx[:, -2] + Fx[:, -3]) / (2*dx))

        dFydy = dFydy.at[1:-1, :].set((Fy[2:, :] - Fy[:-2, :]) / (2 * dy))
        dFydy = dFydy.at[0, :].set((-3*Fy[0, :] + 4*Fy[1, :] - Fy[2, :]) / (2*dy))
        dFydy = dFydy.at[-1, :].set((3*Fy[-1, :] - 4*Fy[-2, :] + Fy[-3, :]) / (2*dy))

        return dFxdx + dFydy
    
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
        d2udx2 = d2udx2.at[:, 0].set((2.0*u[:, 0] - 5.0*u[:, 1] + 4.0*u[:, 2] - u[:, 3]) / (dx ** 2))
        d2udx2 = d2udx2.at[:, -1].set((2.0*u[:, -1] - 5.0*u[:, -2] + 4.0*u[:, -3] - u[:, -4]) / (dx ** 2))

        d2udy2 = d2udy2.at[1:-1, :].set((u[2:, :] - 2 * u[1:-1, :] + u[:-2, :]) / (dy ** 2))
        d2udy2 = d2udy2.at[0, :].set((2.0*u[0, :] - 5.0*u[1, :] + 4.0*u[2, :] - u[3, :]) / (dy ** 2))
        d2udy2 = d2udy2.at[-1, :].set((2.0*u[-1, :] - 5.0*u[-2, :] + 4.0*u[-3, :] - u[-4, :]) / (dy ** 2))

        return d2udx2 + d2udy2


class Losses:
    @classmethod
    def mse_loss(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 **kwargs) -> jnp.ndarray:
        y_pred = vmap(model.forward)(xs)
        # loss = jnp.mean((y_pred - ys) ** 2)
        phi_pred = y_pred[:, 0, :, :]
        phi_true = ys[:, 0, :, :]
        T_pred = y_pred[:, 1, :, :]
        T_true = ys[:, 1, :, :]
        loss_phi = jnp.mean((phi_pred - phi_true) ** 2)
        loss_T = jnp.mean((T_pred - T_true) ** 2)
        loss = loss_phi + loss_T * 4.0
        # phi in (-1, 1)
        # T in (-0.6, 0.05)
        # So the T loss is scaled up by 4.0 to balance the two components
        return loss, {}
    
    @classmethod
    def mse_loss_weighted(cls,
                 model: AutoRegressiveModel2d,
                 xs: jnp.ndarray,
                 ys: jnp.ndarray,
                 dx: float,
                 dy: float,
                 dt: float,
                 **kwargs) -> jnp.ndarray:
        
        def residual_fn(x, y, dx, dy, dt):
            pred = model.forward(x)
            phi = pred[0, :, :]
            nabla_phi = FDM2d.nabla(phi, dx, dy)  # shape: (2, H, W)
            grad_phi2 = jnp.sum(nabla_phi**2, axis=0)  # shape: (H, W)
            weight = 1 + jnp.tanh(grad_phi2)* 10.0
            weight = jax.lax.stop_gradient(weight)
            return weight * (pred - y)
        
        residuals = vmap(residual_fn, in_axes=(0, 0, None, None, None))(xs, ys, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}

    

    @staticmethod
    def kappa(gradx, grady, sigma, eps=1e-12):
        norm_sq = gradx**2 + grady**2 + eps
        cos2theta = (gradx**2 - grady**2) / norm_sq
        sin2theta = 2 * gradx * grady / norm_sq
        cos4theta = cos2theta**2 - sin2theta**2
        return 1 + sigma * cos4theta
    
    @staticmethod
    def H(gradx, grady, sigma, eps=1e-6):
        norm_sq = gradx**2 + grady**2
        
        # 只有当梯度模长足够大时才计算各向异性项
        # 否则在平坦区域，除以极小的 norm6 会导致梯度爆炸或数值噪声
        valid_mask = norm_sq > 1e-3
        
        norm6 = norm_sq**3 + eps
        coef = 16.0 * sigma / norm6
        Hx = coef * gradx * (gradx**2 * grady**2 - grady**4)
        Hy = coef * grady * (grady**2 * gradx**2 - gradx**4)
        
        # 使用 where 过滤掉噪声
        Hx = jnp.where(valid_mask, Hx, 0.0)
        Hy = jnp.where(valid_mask, Hy, 0.0)
        
        return jnp.stack([Hx, Hy], axis=0)

    @classmethod
    def ac_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                ks: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:
        """
        Allen-Cahn equation loss for 2D corrosion modeling.
        """

        def residual_fn(x, k, dx, dy, dt):
            pred = model.forward(x)

            phi0 = x[0, :, :]
            T0 = x[1, :, :]

            phi = pred[0, :, :]
            T = pred[1, :, :]

            dphi_dt = (phi - phi0) / dt / configs.Tc
            nabla_phi = FDM2d.nabla(phi, dx, dy) / configs.Lc # shape: (2, H, W)
            grad_phi_x = nabla_phi[0, :, :]
            grad_phi_y = nabla_phi[1, :, :]
            grad_phi_2 = grad_phi_x**2 + grad_phi_y**2
            kappa_val = cls.kappa(grad_phi_x, grad_phi_y, configs.sigma) # shape: (H, W)
            H = cls.H(grad_phi_x, grad_phi_y, configs.sigma) # shape: (2, H, W)
            # if necessary, stop gradient on kappa
            vec_field = kappa_val**2 * nabla_phi + kappa_val * grad_phi_2 * H  # shape: (2, H, W)
            div_term = FDM2d.divergence(vec_field, dx, dy) / configs.Lc # shape: (H, W)
            F_prime = phi**3 - phi  # shape: (H, W)
            h_prime = phi**4 - 2 * phi**2 + 1  # shape: (H, W)
            r"""
            \frac{\delta E}{\delta \phi} = -\nabla\cdot(\kappa^2 \nabla \phi + \kappa \lvert\nabla \phi\rvert^2 H) + \frac{F'(\phi)}{\epsilon^2})
            """
            residual = (
                configs.rho_val * dphi_dt
                - div_term
                + F_prime / (configs.epsilon ** 2)
                + configs.lam / configs.epsilon * h_prime * T
            )

            return residual / configs.AC_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, 0, None, None, None))(xs, ks, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    
    @classmethod
    def tem_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                ks: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:
        """
        Temperature equation loss for 2D corrosion modeling.
        """

        def residual_fn(x, k, dx, dy, dt):
            pred = model.forward(x)

            phi0 = x[0, :, :]
            T0 = x[1, :, :]

            phi = pred[0, :, :]
            T = pred[1, :, :]
            dT_dt = (T - T0) / dt / configs.Tc
            lap_T = FDM2d.laplacian(T, dx, dy) / configs.Lc**2
            h_prime = phi**4 - 2 * phi**2 + 1  # shape: (H, W)
            dphi_dt = (phi - phi0) / dt / configs.Tc
            residual = (
                dT_dt - configs.D_val * lap_T - k * h_prime * dphi_dt
            )

            return residual / configs.TEM_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, 0, None, None, None))(xs, ks, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    
    @classmethod
    def ac_tem_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                ks: jnp.ndarray,
                dx: float,
                dy: float,
                dt: float,
                configs: Configs,
                **kwargs) -> jnp.ndarray:

        """
        Merge Allen-Cahn and Temperature equation losses using the same dphi/dt term
        """
        
        def residual_fn(x, k, dx, dy, dt):
            pred = model.forward(x)

            phi0 = x[0, :, :]
            T0 = x[1, :, :]

            phi = pred[0, :, :]
            T = pred[1, :, :]

            dphi_dt = (phi - phi0) / dt / configs.Tc
            dT_dt = (T - T0) / dt / configs.Tc
            nabla_phi = FDM2d.nabla(phi, dx, dy) / configs.Lc # shape: (2, H, W)
            grad_phi_x = nabla_phi[0, :, :]
            grad_phi_y = nabla_phi[1, :, :]
            grad_phi_2 = grad_phi_x**2 + grad_phi_y**2
            kappa_val = cls.kappa(grad_phi_x, grad_phi_y, configs.sigma) # shape: (H, W)
            H = cls.H(grad_phi_x, grad_phi_y, configs.sigma) # shape: (2, H, W)
            # if necessary, stop gradient on kappa
            vec_field = kappa_val**2 * nabla_phi + kappa_val * grad_phi_2 * H  # shape: (2, H, W)
            div_term = FDM2d.divergence(vec_field, dx, dy) / configs.Lc # shape: (H, W)
            F_prime = phi**3 - phi  # shape: (H, W)
            h_prime = phi**4 - 2 * phi**2 + 1  # shape: (H, W)

            lap_T = FDM2d.laplacian(T, dx, dy) / configs.Lc**2

            residual = (
                dT_dt - configs.D_val * lap_T 
                + k * h_prime / configs.rho_val * (-div_term)
                + k * h_prime / configs.rho_val * (F_prime / (configs.epsilon ** 2))
                + k * h_prime / configs.rho_val * (configs.lam / configs.epsilon * h_prime * T)
            )

            return residual / configs.TEM_PRE_SCALE
        
        residuals = vmap(residual_fn, in_axes=(0, 0, None, None, None))(xs, ks, dx, dy, dt)
        loss = jnp.mean(jnp.square(residuals))
        return loss, {}
    

    @classmethod
    @eqx.filter_jit
    def pi_loss(cls,
                model: AutoRegressiveModel2d,
                xs: jnp.ndarray,
                ys: jnp.ndarray,
                ks: jnp.ndarray,
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
                model, xs, ys=ys, ks=ks, dx=dx, dy=dy, dt=dt, configs=configs
            )
            losses.append(loss)
            grads.append(grad)
            aux_vars.update(aux_var)

        weights = cls.grad_norm_weights(grads)
        # Adjust weights based on the PDE being solved
        # if pde_name == 'ac':
        #     weights = jnp.array([weights[0], weights[1], 0.0])
        # elif pde_name == 'ch':
        #     weights = jnp.array([weights[0], 0.0, weights[2]])
        # else:
        #     pass
        # weights = weights.at[1].set(weights[1] / 5.0) # Scale down the AC loss weight
        # weights = weights.at[2].set(weights[2] / 5.0) # Scale down the TEM loss weight
    
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
AC_VG  = eqx.filter_value_and_grad(Losses.ac_loss, has_aux=True)
TEM_VG  = eqx.filter_value_and_grad(Losses.tem_loss, has_aux=True)

VG_FNS = [MSE_VG, AC_VG, TEM_VG,]