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


# class FDM2d:
#     """
#     Finite Difference Method for 2D Corrosion Modeling.
#     4th-order accurate (O(h^4)) 5-point stencils with one-sided 5-point
#     boundary formulas for both first and second derivatives.
#     """
#     @staticmethod
#     @eqx.filter_jit
#     def nabla(
#         u: jnp.ndarray,
#         dx: float,
#         dy: float
#     ) -> jnp.ndarray:
#         dudx = jnp.zeros_like(u)
#         dudy = jnp.zeros_like(u)

#         # ---- x-derivative (columns) ----
#         # interior: 5-point central (indices 2 .. -3) -> slice 2:-2
#         # formula: (f_{i-2} - 8 f_{i-1} + 8 f_{i+1} - f_{i+2}) / (12 h)
#         dudx = dudx.at[:, 2:-2].set(
#             (u[:, 0:-4] - 8.0 * u[:, 1:-3] + 8.0 * u[:, 3:-1] - u[:, 4:]) / (12.0 * dx)
#         )

#         # left boundary i=0 (one-sided 5-point, 4th-order)
#         # coeffs: [-25/12, 4, -3, 4/3, -1/4] / dx  applied to u[:,0..4]
#         dudx = dudx.at[:, 0].set(
#             ((-25.0/12.0) * u[:, 0] + 4.0 * u[:, 1] + (-3.0) * u[:, 2] +
#              (4.0/3.0) * u[:, 3] + (-1.0/4.0) * u[:, 4]) / dx
#         )
#         # second column i=1 (one-sided with offsets [-1,0,1,2,3])
#         # coeffs: [-1/4, -5/6, 3/2, -1/2, 1/12] / dx  applied to u[:,0..4] but mapping to [i-1..i+3]
#         dudx = dudx.at[:, 1].set(
#             ((-1.0/4.0) * u[:, 0] + (-5.0/6.0) * u[:, 1] + (3.0/2.0) * u[:, 2] +
#              (-1.0/2.0) * u[:, 3] + (1.0/12.0) * u[:, 4]) / dx
#         )

#         # right boundary symmetric: i = -1 and i = -2
#         dudx = dudx.at[:, -1].set(
#             ((25.0/12.0) * u[:, -1] + (-4.0) * u[:, -2] + 3.0 * u[:, -3] +
#              (-4.0/3.0) * u[:, -4] + (1.0/4.0) * u[:, -5]) / dx
#         )
#         # second-last column i = -2, use offsets [-3,-2,-1,0,1] relative to i=-2
#         # coeffs for i = N-2 (derivative): [-1/12, 1/2, -3/2, 5/6, 1/4] applied to u[:, -5:-0]
#         dudx = dudx.at[:, -2].set(
#             ((-1.0/12.0) * u[:, -5] + (1.0/2.0) * u[:, -4] + (-3.0/2.0) * u[:, -3] +
#              (5.0/6.0) * u[:, -2] + (1.0/4.0) * u[:, -1]) / dx
#         )

#         # ---- y-derivative (rows) ----
#         # interior rows 2:-2
#         dudy = dudy.at[2:-2, :].set(
#             (u[0:-4, :] - 8.0 * u[1:-3, :] + 8.0 * u[3:-1, :] - u[4:, :]) / (12.0 * dy)
#         )

#         # top boundary j=0
#         dudy = dudy.at[0, :].set(
#             ((-25.0/12.0) * u[0, :] + 4.0 * u[1, :] + (-3.0) * u[2, :] +
#              (4.0/3.0) * u[3, :] + (-1.0/4.0) * u[4, :]) / dy
#         )
#         # second row j=1
#         dudy = dudy.at[1, :].set(
#             ((-1.0/4.0) * u[0, :] + (-5.0/6.0) * u[1, :] + (3.0/2.0) * u[2, :] +
#              (-1.0/2.0) * u[3, :] + (1.0/12.0) * u[4, :]) / dy
#         )

#         # bottom boundary symmetric j = -1 and j = -2
#         dudy = dudy.at[-1, :].set(
#             ((25.0/12.0) * u[-1, :] + (-4.0) * u[-2, :] + 3.0 * u[-3, :] +
#              (-4.0/3.0) * u[-4, :] + (1.0/4.0) * u[-5, :]) / dy
#         )
#         dudy = dudy.at[-2, :].set(
#             ((-1.0/12.0) * u[-5, :] + (1.0/2.0) * u[-4, :] + (-3.0/2.0) * u[-3, :] +
#              (5.0/6.0) * u[-2, :] + (1.0/4.0) * u[-1, :]) / dy
#         )

#         return jnp.stack([dudx, dudy], axis=0)

#     @staticmethod
#     @eqx.filter_jit
#     def laplacian(
#         u: jnp.ndarray,
#         dx: float,
#         dy: float
#     ) -> jnp.ndarray:
#         d2udx2 = jnp.zeros_like(u)
#         d2udy2 = jnp.zeros_like(u)

#         # ---- x-second derivative ----
#         # interior (5-point central 4th-order): (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2}) / (12 h^2)
#         d2udx2 = d2udx2.at[:, 2:-2].set(
#             (-u[:, 0:-4] + 16.0 * u[:, 1:-3] - 30.0 * u[:, 2:-2]
#              + 16.0 * u[:, 3:-1] - u[:, 4:]) / (12.0 * dx * dx)
#         )

#         # left boundary i=0 (one-sided 5-point for second derivative)
#         # coeffs: [35/12, -26/3, 19/2, -14/3, 11/12] / dx^2 applied to u[:,0..4]
#         d2udx2 = d2udx2.at[:, 0].set(
#             ((35.0/12.0) * u[:, 0] + (-26.0/3.0) * u[:, 1] + (19.0/2.0) * u[:, 2] +
#              (-14.0/3.0) * u[:, 3] + (11.0/12.0) * u[:, 4]) / (dx * dx)
#         )
#         # second column i=1 (one-sided with offsets [-1,0,1,2,3])
#         # coeffs: [11/12, -5/3, 1/2, 1/3, -1/12] / dx^2
#         d2udx2 = d2udx2.at[:, 1].set(
#             ((11.0/12.0) * u[:, 0] + (-5.0/3.0) * u[:, 1] + (1.0/2.0) * u[:, 2] +
#              (1.0/3.0) * u[:, 3] + (-1.0/12.0) * u[:, 4]) / (dx * dx)
#         )

#         # right boundary symmetric for i = -1 and i = -2
#         d2udx2 = d2udx2.at[:, -1].set(
#             ((35.0/12.0) * u[:, -1] + (-26.0/3.0) * u[:, -2] + (19.0/2.0) * u[:, -3] +
#              (-14.0/3.0) * u[:, -4] + (11.0/12.0) * u[:, -5]) / (dx * dx)
#         )
#         d2udx2 = d2udx2.at[:, -2].set(
#             ((11.0/12.0) * u[:, -5] + (-5.0/3.0) * u[:, -4] + (1.0/2.0) * u[:, -3] +
#              (1.0/3.0) * u[:, -2] + (-1.0/12.0) * u[:, -1]) / (dx * dx)
#         )

#         # ---- y-second derivative ----
#         d2udy2 = d2udy2.at[2:-2, :].set(
#             (-u[0:-4, :] + 16.0 * u[1:-3, :] - 30.0 * u[2:-2, :]
#              + 16.0 * u[3:-1, :] - u[4:, :]) / (12.0 * dy * dy)
#         )

#         d2udy2 = d2udy2.at[0, :].set(
#             ((35.0/12.0) * u[0, :] + (-26.0/3.0) * u[1, :] + (19.0/2.0) * u[2, :] +
#              (-14.0/3.0) * u[3, :] + (11.0/12.0) * u[4, :]) / (dy * dy)
#         )
#         d2udy2 = d2udy2.at[1, :].set(
#             ((11.0/12.0) * u[0, :] + (-5.0/3.0) * u[1, :] + (1.0/2.0) * u[2, :] +
#              (1.0/3.0) * u[3, :] + (-1.0/12.0) * u[4, :]) / (dy * dy)
#         )

#         d2udy2 = d2udy2.at[-1, :].set(
#             ((35.0/12.0) * u[-1, :] + (-26.0/3.0) * u[-2, :] + (19.0/2.0) * u[-3, :] +
#              (-14.0/3.0) * u[-4, :] + (11.0/12.0) * u[-5, :]) / (dy * dy)
#         )
#         d2udy2 = d2udy2.at[-2, :].set(
#             ((11.0/12.0) * u[-5, :] + (-5.0/3.0) * u[-4, :] + (1.0/2.0) * u[-3, :] +
#              (1.0/3.0) * u[-2, :] + (-1.0/12.0) * u[-1, :]) / (dy * dy)
#         )

#         return d2udx2 + d2udy2


# class Spectral2d:
#     """
#     Fourier spectral differentiation for 2D fields.
#     """

#     @staticmethod
#     def _get_kx_ky(nx: int, ny: int, dx: float = 1.0, dy: float = 1.0):
#         # Frequencies for FFT domain
#         # 在FNO模型中，我们用的是 nx, ny 的顺序
#         # 但是在mesh中，实际上的顺序是 y, x
#         # 所以这里的 kx, ky 实际上是对应 y 方向和 x 方向的频率
#         kx = jnp.fft.fftfreq(nx, d=dy) * (2 * jnp.pi)
#         ky = jnp.fft.rfftfreq(ny, d=dx) * (2 * jnp.pi)
#         return kx, ky

#     # ---------------------------
#     #       ∇u = (ux, uy)
#     # ---------------------------
#     @staticmethod
#     @eqx.filter_jit
#     def nabla(u: jnp.ndarray, dx, dy) -> jnp.ndarray:
#         """
#         Compute gradient of u using Fourier spectral differentiation.
#         u shape: (nx, ny) OR (channels, nx, ny)
#         Returns: (2, nx, ny)   # [ux, uy]
#         """
#         # Support both single-field and multi-channel input
#         if u.ndim == 2:
#             u = u[None, ...]   # add channel dim

#         C, nx, ny = u.shape
#         kx, ky = Spectral2d._get_kx_ky(nx, ny, dx, dy)

#         # FFT of u
#         u_hat = jnp.fft.rfftn(u, axes=(-2, -1))           # (C, nx, ny//2+1)

#         # reshape frequency grids
#         KX = kx.reshape(1, nx, 1)
#         KY = ky.reshape(1, 1, ny//2 + 1)

#         # spectral derivatives
#         # KX 对应 y 方向导数，KY 对应 x 方向导数
#         ux_hat = 1j * KY * u_hat  # du/dx uses kx (KY)
#         uy_hat = 1j * KX * u_hat  # du/dy uses ky (KX)

#         # inverse FFT to get spatial derivatives
#         ux = jnp.fft.irfftn(ux_hat, s=(nx, ny), axes=(-2, -1))
#         uy = jnp.fft.irfftn(uy_hat, s=(nx, ny), axes=(-2, -1))

#         # Return (2, nx, ny) regardless of channels
#         if C == 1:
#             return jnp.stack([ux[0], uy[0]], axis=0)
#         else:
#             return jnp.stack([ux, uy], axis=0)    # shape (2, C, nx, ny)


#     # ---------------------------
#     #       ∇²u = u_xx + u_yy
#     # ---------------------------
#     @staticmethod
#     @eqx.filter_jit
#     def laplacian(u: jnp.ndarray, dx, dy) -> jnp.ndarray:
#         """
#         Compute Laplacian using Fourier spectral differentiation.
#         u shape: (nx, ny) OR (channels, nx, ny)
#         Returns: (nx, ny) OR (channels, nx, ny)
#         """
#         # Normalize shape
#         if u.ndim == 2:
#             u = u[None, ...]

#         C, nx, ny = u.shape
#         kx, ky = Spectral2d._get_kx_ky(nx, ny, dx, dy)

#         u_hat = jnp.fft.rfftn(u, axes=(-2, -1))

#         KX = kx.reshape(1, nx, 1)
#         KY = ky.reshape(1, 1, ny//2 + 1)

#         lap_hat = -(KX**2 + KY**2) * u_hat

#         lap = jnp.fft.irfftn(lap_hat, s=(nx, ny), axes=(-2, -1))

#         if C == 1:
#             return lap[0]
#         return lap




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
        loss = jnp.mean(jnp.square(residuals))
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
        loss = jnp.mean(jnp.square(residuals))
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
            # left = (u[:, 1] - u[:, 0])
            # right = (u[:, -1] - u[:, -2])
            left = ( -3 * u[:, 0] + 4 * u[:, 1] - u[:, 2] ) / (2 * dx)
            right = ( 3 * u[:, -1] - 4 * u[:, -2] + u[:, -3] ) / (2 * dx)
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
# BC_VG  = eqx.filter_value_and_grad(Losses.bc_loss, has_aux=True)
VG_FNS = [MSE_VG, AC_VG, CH_VG,]