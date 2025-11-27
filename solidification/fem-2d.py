from dolfin import *
import numpy as np
import time
import os

# 优化编译器参数
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
set_log_level(LogLevel.ERROR)

# ==================== 模式设置 ====================
mode = 'train_valid'
# mode = 'test'

if mode == 'train_valid':
    K_values = [0.6,0.7, 0.8,0.9, 1.0, 1.1,1.2]
    save_dir = './solidification/data/train_valid'
elif mode == 'test':
    K_values = [0.65, 0.75, 0.85, 0.95, 1.05, 1.15]
    save_dir = './solidification/data/test'
else:
    raise ValueError("Invalid mode")

if not os.path.exists(save_dir):
    os.makedirs(save_dir)
K_values = np.array(K_values)
np.save(f'{save_dir}/K_values.npy', K_values)

# ==================== 参数设置 ====================
# 模型参数 (保持不变)
rho_val = 1e3        
eps = 0.015          
sigma = 0.1          
lam = 4e2            
D_val = 2.5e-3       
# K_val 将在循环中动态赋值

# 数值参数 (保持不变)
Nx, Ny = 128, 128    
T_final = 8        
dt = 0.05           
num_steps = int(T_final / dt)

# 初始条件参数
r0 = 0.05
x0, y0 = 0.0, 0.0
eps0 = eps

# ==================== 网格和函数空间 ====================
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx, Ny, 'crossed')
V = FunctionSpace(mesh, 'P', 1)

# 保存网格节点坐标（原始DOF坐标）
mesh_points = V.tabulate_dof_coordinates()
print(f"Mesh points shape: {mesh_points.shape}")
np.save(f'{save_dir}/mesh_points.npy', mesh_points)

# 生成规则网格坐标（仿照corrosion2d方法）
x_coords = np.linspace(-1, 1, Nx + 1)
y_coords = np.linspace(-1, 1, Ny + 1)
X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
grid_points = np.vstack([X.ravel(), Y.ravel()]).T
# 保存规则网格坐标
np.save(f'{save_dir}/mesh_grid_coords.npy', grid_points.reshape((Ny + 1, Nx + 1, 2)))

num_points = mesh_points.shape[0]
num_grid_points = grid_points.shape[0]

# ==================== 函数定义 ====================
phi = Function(V)
T_var = Function(V)
phi_n = Function(V)
T_n = Function(V)

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

# ==================== 模型函数 ====================
def F_potential(phi):
    return 0.25 * (phi**2 - 1)**2

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

# ==================== 求解器 ====================
# 注意：这里需要传入当前的 K_val
def solve_direct(current_K):
    v_phi = TestFunction(V)
    v_T = TestFunction(V)
    phi_trial = TrialFunction(V)
    T_trial = TrialFunction(V)
    
    grad_phi_n = grad(phi_n)
    kappa_n = kappa(grad_phi_n)
    H_n = compute_H(grad_phi_n)
    
    # 方程 1: Phase Field
    df_dphi_semi = (phi_n**2 - 1) * phi_trial
    F_phi = (
        rho_val * (phi_trial - phi_n) / dt * v_phi * dx
        + kappa_n**2 * dot(grad(phi_trial), grad(v_phi)) * dx
        + kappa_n * dot(grad_phi_n, grad_phi_n) * dot(H_n, grad(v_phi)) * dx
        + df_dphi_semi / eps**2 * v_phi * dx
        + (lam / eps) * h_prime(phi_n) * T_n * v_phi * dx
    )
    
    solve(lhs(F_phi) == rhs(F_phi), phi, 
          solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"})
    
    # 方程 2: Temperature (使用传入的 current_K)
    F_T = (
        (T_trial - T_n) / dt * v_T * dx
        + D_val * dot(grad(T_trial), grad(v_T)) * dx
        - current_K * h_prime(phi_n) * (phi - phi_n) / dt * v_T * dx
    )
    
    solve(lhs(F_T) == rhs(F_T), T_var, 
          solver_parameters={"linear_solver": "gmres", "preconditioner": "ilu"})
    
    return phi, T_var

# ==================== 主循环 ====================
# 初始化结果数组: [num_K, num_time_steps, num_vars, num_nodes]
# num_vars = 2 (phi, T)
results = np.zeros((len(K_values), num_steps + 1, 2, num_points))
# 添加规则网格结果数组: [num_K, num_time_steps, num_vars, ny+1, nx+1]
results_grid = np.zeros((len(K_values), num_steps + 1, 2, Ny + 1, Nx + 1))
times = np.linspace(0, T_final, num_steps + 1)

print("="*70)
print(f"开始计算 Mode: {mode}")
print(f"K values: {K_values}")
print(f"Results shape: {results.shape}")
print(f"Results grid shape: {results_grid.shape}")
print("="*70)

total_start_time = time.time()

for k_idx, K_val in enumerate(K_values):
    print(f"\nStarting simulation with K = {K_val} ({k_idx+1}/{len(K_values)})")
    
    # 1. 重置初始条件
    phi_n.interpolate(PhiInitial(degree=2))
    T_n.interpolate(TInitial(degree=2))
    
    # 2. 存储初始时刻 (t=0)
    phi_vec = phi_n.vector().get_local()
    T_vec = T_n.vector().get_local()
    results[k_idx, 0, 0, :] = phi_vec
    results[k_idx, 0, 1, :] = T_vec
    
    # 在规则网格上采样初始条件并存储
    sampled_init = np.array([(phi_n(float(px), float(py)), T_n(float(px), float(py))) for px, py in grid_points])
    # sampled_init shape -> (num_grid, 2) ; reshape to (2, ny+1, nx+1)
    results_grid[k_idx, 0] = sampled_init.reshape((Ny + 1, Nx + 1, 2)).transpose(2, 0, 1)
    
    # 3. 时间步进
    t = 0.0
    case_start_time = time.time()
    
    for n in range(num_steps):
        t += dt
        
        # 求解
        phi, T_var = solve_direct(K_val)
        
        # 存储结果
        phi_vec = phi.vector().get_local()
        T_vec = T_var.vector().get_local()
        results[k_idx, n + 1, 0, :] = phi_vec
        results[k_idx, n + 1, 1, :] = T_vec
        
        # 在规则网格上采样解决方案并存储
        sampled = np.array([(phi(float(px), float(py)), T_var(float(px), float(py))) for px, py in grid_points])
        # sampled.shape == (num_grid, 2)
        results_grid[k_idx, n + 1] = sampled.reshape((Ny + 1, Nx + 1, 2)).transpose(2, 0, 1)
        
        # 更新历史值
        phi_n.assign(phi)
        T_n.assign(T_var)
        
        # 打印进度
        if (n + 1) % 50 == 0:
            print(f"  Step {n+1}/{num_steps}, t={t:.3f}")

    print(f"Completed K = {K_val}, Time: {time.time() - case_start_time:.1f}s")

# ==================== 保存结果 ====================
print("="*70)
print("Saving results...")

np.save(f'{save_dir}/solutions.npy', results)
np.save(f'{save_dir}/solutions_grid.npy', results_grid)
np.save(f'{save_dir}/K_values.npy', K_values)
np.save(f'{save_dir}/times.npy', times)

print(f"Results saved to {save_dir}")
print(f"Original solutions shape: {results.shape}")
print(f"Grid solutions shape: {results_grid.shape}")