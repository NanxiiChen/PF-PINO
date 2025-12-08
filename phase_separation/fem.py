"""
Cahn-Hilliard 相分离问题的 FEniCS 实现
使用混合有限元方法求解
"""

from fenics import *
import numpy as np
import os
import os

os.environ['OMP_NUM_THREADS'] = '16'
print(f"Current OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

# 设置模式和保存路径
# mode = 'train_valid'
mode = 'test'
save_dir = './phase_separation/data/train_valid' if mode == 'train_valid' else './phase_separation/data/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置参数
lx, ly = 1.0, 1.0  # 区域尺寸
nx, ny = 64, 64  # 网格数
dt = 5.0e-5        # 时间步长
theta = 0.5        # 时间离散参数 (Crank-Nicolson)
T = 0.01           # 总时间
num_steps = int(T / dt)

# Cahn-Hilliard 参数
M = 1.0            # 迁移率
epsilon = 0.01     # 界面厚度参数
lambda_param = epsilon**2

# 创建网格
mesh = RectangleMesh(Point(0, 0), Point(lx, ly), nx, ny)

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

# 设置随机初始条件 (修改为接受种子)
class InitialConditions(UserExpression):
    def __init__(self, seed, **kwargs):
        super().__init__(**kwargs)
        self.seed = seed
        np.random.seed(seed)
    
    def eval(self, values, x):
        values[0] = 0.0 + 0.05 * (0.5 - np.random.rand())  # c
        values[1] = 0.0  # mu
    
    def value_shape(self):
        return (2,)

# 双势阱自由能的导数: f'(c) = c^3 - c
def dfdc(c):
    return c**3 - c

# 时间离散的浓度
c_mid = (1.0 - theta) * c0 + theta * c

# 弱形式
# 方程1: dc/dt = div(M * grad(mu))
L0 = (c - c0) * q_c * dx + dt * M * dot(grad(mu), grad(q_c)) * dx

# 方程2: mu = df/dc - lambda * laplacian(c)
L1 = mu * q_mu * dx - dfdc(c_mid) * q_mu * dx - lambda_param * dot(grad(c_mid), grad(q_mu)) * dx

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
num_initials = 10 if mode == 'train_valid' else 5
initial_seeds = [100 + i for i in range(num_initials)] \
    if mode == 'train_valid' \
    else [200 + i for i in range(num_initials)]

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
    
    # 初始化
    u_init = InitialConditions(seed=seed, degree=1)
    u.interpolate(u_init)
    u0.interpolate(u_init)
    
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
np.save(os.path.join(save_dir, 'solutions_initials.npy'), results)
np.save(os.path.join(save_dir, 'solutions_grid_initials.npy'), results_grid)
np.save(os.path.join(save_dir, 'initial_seeds.npy'), np.array(initial_seeds))
np.save(os.path.join(save_dir, 'times.npy'), times)

print(f"\n模拟完成! 结果已保存至 {save_dir}")
print(f"Grid shape: {results_grid.shape}")