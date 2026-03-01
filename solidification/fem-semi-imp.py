from dolfin import *
import numpy as np
import time
import os
import argparse

# 设置并行线程数（可选）
os.environ['OMP_NUM_THREADS'] = '20'

print(f"Current OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

# 优化编译器参数
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(LogLevel.ERROR)

# ==================== 模式设置 ====================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_valid', choices=['train_valid', 'test'], help='Select mode: train_valid or test')
args = parser.parse_args()
mode = args.mode

if mode == 'train_valid':
    K_values = [0.8, 1.0, 1.4, 1.6, 2.2]
    save_dir = "/root/autodl-tmp/data/solidification_0210_experiments/train_valid/"
elif mode == 'test':
    K_values = [0.9, 1.3, 1.7, 2.0]
    save_dir = "./data/solidification_1220/test/"
else:
    raise ValueError("Invalid mode")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
K_values = np.array(K_values)
np.save(f'{save_dir}/K_values.npy', K_values)

# ==================== 参数设置 ====================
# 模型参数
rho_val = 1e3        
eps = 0.015          
sigma = 0.1          
lam = 4e2            
D_val = 2.5e-3       

# 数值参数
Nx, Ny = 128, 128    
T_final = 10       
dt = 0.01          # 线性化隐式方法通常允许比显式更大的 dt，但为了精度对比保持 0.01
save_dt = 0.05
save_every = int(save_dt / dt)
num_steps = int(T_final / dt)
num_save_steps = num_steps // save_every + 1

# 初始条件参数
r0 = 0.05
x0, y0 = 0.0, 0.0
eps0 = eps

# ==================== 网格和函数空间 ====================
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx, Ny, 'crossed')
V = FunctionSpace(mesh, 'P', 1) # 回归独立空间

# 保存网格节点坐标
mesh_points = V.tabulate_dof_coordinates()
print(f"Mesh points shape: {mesh_points.shape}")

# 生成规则网格坐标
x_coords = np.linspace(-1, 1, Nx + 1)
y_coords = np.linspace(-1, 1, Ny + 1)
X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
grid_points = np.vstack([X.ravel(), Y.ravel()]).T
np.save(f'{save_dir}/mesh_grid_coords.npy', grid_points.reshape((Ny + 1, Nx + 1, 2)))

num_points = mesh_points.shape[0]

# ==================== 函数定义 ====================
phi = Function(V)
phi_n = Function(V)
T_val = Function(V) # 避免与 T 符号冲突
T_n = Function(V)

phi_trial = TrialFunction(V)
v_phi = TestFunction(V)
T_trial = TrialFunction(V)
v_T = TestFunction(V)

# ==================== 初始条件表达式 ====================
class PhiInitial(UserExpression):
    def eval(self, values, x):
        r = np.sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        values[0] = np.tanh((r0 - r) / eps0)

class TInitial(UserExpression):
    def eval(self, values, x):
        r = np.sqrt((x[0]-x0)**2 + (x[1]-y0)**2)
        phi_val = np.tanh((r0 - r) / eps0)
        if phi_val > 0:
            values[0] = 0.0
        else:
            values[0] = -0.6

# ==================== 辅助函数 ====================
def h_prime(phi):
    return phi**4 - 2*phi**2 + 1

def kappa(grad_phi):
    eps_reg = 1e-12
    phi_x, phi_y = grad_phi[0], grad_phi[1]
    norm_sq = phi_x**2 + phi_y**2 + eps_reg
    cos2theta = (phi_x**2 - phi_y**2) / norm_sq
    sin2theta = 2 * phi_x * phi_y / norm_sq
    cos4theta = cos2theta**2 - sin2theta**2
    return 1 + sigma * cos4theta

def compute_H(grad_phi):
    phi_x, phi_y = grad_phi[0], grad_phi[1]
    norm6 = (phi_x**2 + phi_y**2)**3 + 1e-18
    coef = 16.0 * sigma / norm6
    Hx = coef * phi_x * (phi_x**2 * phi_y**2 - phi_y**4)
    Hy = coef * phi_y * (phi_y**2 * phi_x**2 - phi_x**4)
    return as_vector((Hx, Hy))

# ==================== 变分问题定义 (线性化半隐式) ====================
K_const = Constant(0.0)
dt_const = Constant(dt)

# --- 1. Phi 方程 (线性化) ---
# 线性化非线性源项 f(phi) = phi - phi^3
# f(phi) approx f(phi_n) + f'(phi_n) * (phi - phi_n)
#        = (phi_n - phi_n^3) + (1 - 3*phi_n^2) * (phi - phi_n)
#        = 2*phi_n^3 + (1 - 3*phi_n^2) * phi
# 移项后：
# LHS: rho/dt * phi - div(kappa^2 grad(phi)) - 1/eps^2 * (1 - 3*phi_n^2) * phi
# RHS: rho/dt * phi_n + 1/eps^2 * 2*phi_n^3 + Coupling + Anisotropy_Explicit

grad_phi_n = grad(phi_n)
kappa_n = kappa(grad_phi_n)
H_n = compute_H(grad_phi_n)

# 左端项 (隐式部分, 包含待求的 phi_trial)
# 注意：kappa 系数使用上一时刻值 (frozen coefficient)，这使得方程变为线性
lhs_phi = (
    rho_val * phi_trial / dt_const * v_phi * dx
    + kappa_n**2 * dot(grad(phi_trial), grad(v_phi)) * dx
    - (1.0 / eps**2) * (1 - 3*phi_n**2) * phi_trial * v_phi * dx
)

# 右端项 (显式部分, 全部已知)
# 注意：各向异性修正项 (H_n) 保持显式，因为它很难线性化且通常较小
rhs_phi = (
    rho_val * phi_n / dt_const * v_phi * dx
    - kappa_n * dot(grad_phi_n, grad_phi_n) * dot(H_n, grad(v_phi)) * dx
    + (1.0 / eps**2) * (2 * phi_n**3) * v_phi * dx
    - (lam / eps) * h_prime(phi_n) * T_n * v_phi * dx
)

# 定义 Phi 求解器
problem_phi = LinearVariationalProblem(lhs_phi, rhs_phi, phi)
solver_phi = LinearVariationalSolver(problem_phi)
solver_phi.parameters["linear_solver"] = "gmres"
solver_phi.parameters["preconditioner"] = "ilu"

# --- 2. T 方程 (隐式) ---
# (T - T_n)/dt = D * laplacian(T) - Source
# Source = K * h'(phi_new) * (phi_new - phi_n) / dt
# 这里 phi_new 已经在第一步求出，所以 Source 是已知的

lhs_T = (
    T_trial / dt_const * v_T * dx
    + D_val * dot(grad(T_trial), grad(v_T)) * dx
)

rhs_T = (
    T_n / dt_const * v_T * dx
    + K_const * h_prime(phi) * (phi - phi_n) / dt_const * v_T * dx
)

# 定义 T 求解器
problem_T = LinearVariationalProblem(lhs_T, rhs_T, T_val)
solver_T = LinearVariationalSolver(problem_T)
solver_T.parameters["linear_solver"] = "gmres"
solver_T.parameters["preconditioner"] = "ilu"


# ==================== 主循环 ====================
results = np.zeros((len(K_values), num_save_steps, 2, num_points))
results_grid = np.zeros((len(K_values), num_save_steps, 2, Ny + 1, Nx + 1))
times = np.linspace(0, T_final, num_steps + 1)
save_times = times[::save_every]

print("="*70)
print(f"开始计算 (Linearized Semi-Implicit) Mode: {mode}")
print(f"K values: {K_values}")
print("="*70)

total_start_time = time.time()

for k_idx, K_val in enumerate(K_values):
    print(f"\nStarting simulation with K = {K_val} ({k_idx+1}/{len(K_values)})")
    
    K_const.assign(K_val)
    
    # 1. 初始化
    phi_n.interpolate(PhiInitial(degree=2))
    T_n.interpolate(TInitial(degree=2))
    
    # 2. 存储初始时刻
    save_idx = 0
    
    results[k_idx, save_idx, 0, :] = phi_n.vector().get_local()
    results[k_idx, save_idx, 1, :] = T_n.vector().get_local()
    
    sampled_init = np.array([(phi_n(float(px), float(py)), T_n(float(px), float(py))) for px, py in grid_points])
    results_grid[k_idx, save_idx] = sampled_init.reshape((Ny + 1, Nx + 1, 2)).transpose(2, 0, 1)
    
    save_idx += 1
    
    # 3. 时间步进
    t = 0.0
    case_start_time = time.time()
    
    for n in range(num_steps):
        t += dt
        
        # Step 1: Solve Phi (Linearized)
        solver_phi.solve()
        
        # Step 2: Solve T (Implicit, using new phi)
        solver_T.solve()
        
        # 保存
        if (n + 1) % save_every == 0:
            results[k_idx, save_idx, 0, :] = phi.vector().get_local()
            results[k_idx, save_idx, 1, :] = T_val.vector().get_local()
            
            sampled = np.array([(phi(float(px), float(py)), T_val(float(px), float(py))) for px, py in grid_points])
            results_grid[k_idx, save_idx] = sampled.reshape((Ny + 1, Nx + 1, 2)).transpose(2, 0, 1)
            
            save_idx += 1
            print(f"  Step {n+1}/{num_steps}, t={t:.3f}")

        # 更新上一时刻解
        phi_n.assign(phi)
        T_n.assign(T_val)

    print(f"Completed K = {K_val}, Time: {time.time() - case_start_time:.1f}s")

# ==================== 保存结果 ====================
print("="*70)
print("Saving results...")

np.save(f'{save_dir}/solutions_grid.npy', results_grid) 
np.save(f'{save_dir}/K_values.npy', K_values)
np.save(f'{save_dir}/times.npy', save_times)

print(f"Results saved to {save_dir}")