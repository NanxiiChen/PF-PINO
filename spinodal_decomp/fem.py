"""
Cahn-Hilliard 相分离问题的 FEniCS 实现
使用混合有限元方法求解
python spinodal_decomp/fem.py --mode train_valid
python spinodal_decomp/fem.py --mode test
python spinodal_decomp/fem.py --mode train_init_steps
"""

from fenics import *
import numpy as np
import os
import os

os.environ['OMP_NUM_THREADS'] = '16'
print(f"Current OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

# 设置模式和保存路径
import argparse
mode = 'train_valid'  # 默认模式
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_valid', choices=['train_valid', 'test', 'train_init_steps'],
                    help='选择数据集模式: train_valid 或 test')
args = parser.parse_args()
mode = args.mode

# save_dir = './spinodal_decomp/data/train_valid' if mode == 'train_valid' else './spinodal_decomp/data/test'
if mode == 'train_valid':
    save_dir = './spinodal_decomp/data/train_valid'
elif mode == 'test':
    save_dir = './spinodal_decomp/data/test'
elif mode == 'train_init_steps':
    save_dir = './spinodal_decomp/data/train_init_steps'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置参数
lx, ly = 1.0, 1.0  # 区域尺寸
nx, ny = 64, 64  # 网格数
dt = 5.0e-5        # 时间步长
theta = 0.5        # 时间离散参数 (Crank-Nicolson)
T = 5e-3           # 总时间
if mode == 'train_init_steps':
    T = dt * 10     # 仅初始步骤
num_steps = int(T / dt)

# Cahn-Hilliard 参数
M = 1.0            # 迁移率
epsilon = 0.01     # 界面厚度参数
lambda_param = epsilon**2

# 创建网格
mesh = RectangleMesh(Point(0, 0), Point(lx, ly), nx, ny)


# def smooth_random_field(x, y, Lx, Ly, modes=5, amp=0.05, seed=None):
#     if seed is not None:
#         np.random.seed(seed)
#     val = 0.0
#     for _ in range(modes):
#         # kx = np.random.randint(1, 3)
#         # ky = np.random.randint(1, 3)
#         kx = np.random.choice([1,2])
#         ky = np.random.choice([1,2])
#         phase = np.random.rand()*2*np.pi
#         A = np.random.randn() * amp
#         val += A * np.sin(2*np.pi*(kx*x/Lx + ky*y/Ly) + phase)
#     return val


# 定义周期性边界条件
class PeriodicBoundary(SubDomain):
    # 左/下边为主边，右/上边为从属边，角点只映射一次
    def inside(self, x, on_boundary):
        return bool(
            (near(x[0], 0.0) or near(x[1], 0.0)) and on_boundary and
            not ((near(x[0], 0.0) and near(x[1], ly)) or (near(x[0], lx) and near(x[1], 0.0)) or (near(x[0], lx) and near(x[1], ly)))
        )

    def map(self, x, y):
        # 将右边界映射到左边界，将上边界映射到下边界
        if near(x[0], lx) and near(x[1], ly):
            y[0] = x[0] - lx
            y[1] = x[1] - ly
        elif near(x[0], lx):
            y[0] = x[0] - lx
            y[1] = x[1]
        elif near(x[1], ly):
            y[0] = x[0]
            y[1] = x[1] - ly
        else:
            y[0] = x[0]
            y[1] = x[1]

pb = PeriodicBoundary()

# 定义混合函数空间 (c, mu) 并施加周期性约束
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, MixedElement([P1, P1]), constrained_domain=pb)

# --- 新增: 定义用于计算初始 mu 的标量空间 ---
V_scalar = FunctionSpace(mesh, P1, constrained_domain=pb)
# -----------------------------------------

# 定义试函数和测试函数
dq = TrialFunction(ME)
q = TestFunction(ME)
q_c, q_mu = split(q)

# 定义当前时刻和上一时刻的解
u = Function(ME)
u0 = Function(ME)

# 分离浓度和化学势
c, mu = split(u)
c0, mu0 = split(u0)

# 设置随机初始条件 (修改为宽频噪声叠加)
class InitialConditions(UserExpression):
    def __init__(self, seed, lx, ly, **kwargs):
        super().__init__(**kwargs)
        self.lx = lx
        self.ly = ly
        np.random.seed(seed)
        
        # --- 噪声配置 ---
        self.num_modes = 200      # 叠加 200 个波，足够产生“乱”的感觉
        self.k_max = 20           # 最大波数设为 20 (对应波长 0.05，接近界面厚度)
        self.noise_amp = 0.1     # 噪声总强度
        
        # --- 预先生成并固定随机系数 ---
        # 随机选择波数 kx, ky 在 [-20, 20] 之间
        self.kx = np.random.randint(-self.k_max, self.k_max + 1, self.num_modes)
        self.ky = np.random.randint(-self.k_max, self.k_max + 1, self.num_modes)
        
        # 随机相位 [0, 2pi]
        self.phases = np.random.rand(self.num_modes) * 2 * np.pi
        
        # 振幅归一化：根据中心极限定理，叠加 N 个波后方差会变大
        # 所以单波振幅要除以 sqrt(N)
        self.scale = self.noise_amp / np.sqrt(self.num_modes)

    def eval(self, values, x):
        val = 0.0
        # 叠加所有模式: sum( A * cos(k*x + phi) )
        # 这里使用简单的循环，对于 200 次循环在 C++ 回调中是可以接受的
        for i in range(self.num_modes):
            kx = self.kx[i]
            ky = self.ky[i]
            phase = self.phases[i]
            
            # 计算波的值
            theta = 2 * np.pi * (kx * x[0] / self.lx + ky * x[1] / self.ly) + phase
            val += np.cos(theta)
            
        values[0] = val * self.scale
        values[1] = 0.0 # mu 初始设为 0，稍后会被投影修正覆盖

    def value_shape(self):
        return (2,)

# 双势阱自由能的导数: f'(c) = c^3 - c
def dfdc(c):
    return c**3 - c


# 弱形式
# 方程1: dc/dt = div(M * grad(mu))
L0 = (c - c0) * q_c * dx + dt * M * dot(grad(mu), grad(q_c)) * dx

# 方程2: mu = df/dc - lambda * laplacian(c)
L1 = mu * q_mu * dx - dfdc(c) * q_mu * dx - lambda_param * dot(grad(c), grad(q_mu)) * dx

# 总弱形式
L = L0 + L1

# 计算雅可比矩阵
a = derivative(L, u, dq)

# 创建非线性问题和求解器
problem = NonlinearVariationalProblem(L, u, J=a)
solver = NonlinearVariationalSolver(problem)

# 求解器参数设置
prm = solver.parameters
prm['newton_solver']['absolute_tolerance'] = 1E-8
prm['newton_solver']['relative_tolerance'] = 1E-7
prm['newton_solver']['maximum_iterations'] = 25

# --- 批量计算与数据保存设置 ---

# 构建规则网格坐标
x_coords = np.linspace(0, lx, nx + 1)
y_coords = np.linspace(0, ly, ny + 1)
X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
grid_points = np.vstack([X.ravel(), Y.ravel()]).T

# 保存网格坐标
np.save(os.path.join(save_dir, 'mesh_grid_coords.npy'), grid_points.reshape((ny + 1, nx + 1, 2)))

# 设置种子列表
# num_initials = 1 if mode == 'train_valid' else 5
if mode == 'train_valid':
    num_initials = 20
elif mode == 'test':
    num_initials = 5
elif mode == 'train_init_steps':
    num_initials = 20
# initial_seeds = [100 + i for i in range(num_initials)] \
#     if mode == 'train_valid' \
#     else [200 + i for i in range(num_initials)]
if mode == 'train_valid':
    initial_seeds = [100 + i for i in range(num_initials)]
elif mode == 'test':
    initial_seeds = [200 + i for i in range(num_initials)]
elif mode == 'train_init_steps':
    initial_seeds = [300 + i for i in range(num_initials)]

# 初始化结果数组
# results: 存储原始DOF (num_initials, steps, 2, num_dofs_per_var)
# results_grid: 存储规则网格插值 (num_initials, steps, 2, ny+1, nx+1)
num_dofs_total = u.vector().size()
num_dofs_per_var = num_dofs_total // 2 # 假设混合单元DOF交错且相等
results = np.zeros((num_initials, num_steps + 1, 2, num_dofs_per_var))
results_grid = np.zeros((num_initials, num_steps + 1, 2, ny + 1, nx + 1))
times = np.linspace(0, T, num_steps + 1)

print(f"开始批量模拟... 模式: {mode}, 样本数: {num_initials}, 总步数: {num_steps}")

for i, seed in enumerate(initial_seeds):
    print(f"正在计算样本 {i+1}/{num_initials}, 种子: {seed}")
    
    # 初始化 (传入 lx, ly)
    u_init = InitialConditions(seed=seed, lx=lx, ly=ly, degree=1)
    u.interpolate(u_init)
    
    # --- 新增: 修复初始 mu (使其与 c 物理一致) ---
    # 此时 u 中的 c 是随机的，mu 是 0。我们需要根据 c 计算一致的 mu。
    # 求解投影问题: (mu, v) = (df/dc, v) + lambda * (grad(c), grad(v))
    mu_proj = TrialFunction(V_scalar)
    v_proj = TestFunction(V_scalar)
    c_curr, _ = split(u) # 获取当前的 c (UFL 表达式)
    
    # 定义投影的弱形式 (分部积分处理拉普拉斯项)
    a_init = mu_proj * v_proj * dx
    L_init = dfdc(c_curr) * v_proj * dx + lambda_param * dot(grad(c_curr), grad(v_proj)) * dx
    
    mu_consistent = Function(V_scalar)
    solve(a_init == L_init, mu_consistent)
    
    # 将计算出的 mu 赋值回 u 的第二个分量
    assign(u.sub(1), mu_consistent)
    # -------------------------------------------

    u0.assign(u) # 确保 u0 也同步更新
    
    # 保存初始状态 (t=0)
    # 1. 保存 DOF
    u_vec = u.vector().get_local()
    results[i, 0, 0, :] = u_vec[0::2] # c
    results[i, 0, 1, :] = u_vec[1::2] # mu
    
    # 2. 保存网格插值
    sampled = np.array([u(Point(px, py)) for px, py in grid_points])
    # sampled shape: (num_grid_points, 2) -> reshape to (ny+1, nx+1, 2) -> transpose to (2, ny+1, nx+1)
    results_grid[i, 0] = sampled.reshape((ny + 1, nx + 1, 2)).transpose(2, 0, 1)
    
    t = 0.0
    # 时间演化
    for step in range(num_steps):
        t += dt
        
        # 求解非线性问题
        try:
            solver.solve()
        except Exception as e:
            print(f"  求解失败: 样本 {i}, 步数 {step}, 错误: {e}")
            break
        
        # 更新上一时刻的解
        u0.assign(u)
        
        # 保存当前步结果
        u_vec = u.vector().get_local()
        results[i, step + 1, 0, :] = u_vec[0::2]
        results[i, step + 1, 1, :] = u_vec[1::2]
        
        sampled = np.array([u(Point(px, py)) for px, py in grid_points])
        results_grid[i, step + 1] = sampled.reshape((ny + 1, nx + 1, 2)).transpose(2, 0, 1)
        
        if (step + 1) % 20 == 0:
            print(f"    完成步数 {step + 1}/{num_steps}")

# 保存所有结果
# np.save(os.path.join(save_dir, 'solutions.npy'), results)
np.save(os.path.join(save_dir, 'solutions_grid.npy'), results_grid)
np.save(os.path.join(save_dir, 'initial_seeds.npy'), np.array(initial_seeds))
np.save(os.path.join(save_dir, 'times.npy'), times)

print(f"\n模拟完成! 结果已保存至 {save_dir}")
print(f"Grid shape: {results_grid.shape}")