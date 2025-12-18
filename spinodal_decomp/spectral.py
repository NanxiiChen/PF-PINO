"""
Cahn-Hilliard 相分离问题的 spectral 方法实现
使用伪谱法求解，保持与 FEM 版本相同的参数设置
python spinodal_decomp/spectral.py --mode train_valid
python spinodal_decomp/spectral.py --mode test
python spinodal_decomp/spectral.py --mode train_init_steps
"""

import numpy as np
import os
import argparse

# 设置模式和保存路径
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train_valid', choices=['train_valid', 'test', 'train_init_steps'],
                    help='选择数据集模式: train_valid 或 test')
args = parser.parse_args()
mode = args.mode

# 为了不覆盖 FEM 的结果，保存到 data_spectral 目录
if mode == 'train_valid':
    save_dir = '/root/autodl-tmp/data/spinodal_decomp_spectra/train_valid'
elif mode == 'test':
    save_dir = '/root/autodl-tmp/data/spinodal_decomp_spectra/test'
elif mode == 'train_init_steps':
    save_dir = '/root/autodl-tmp/data/spinodal_decomp_spectra/train_init_steps'


if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 设置参数 (与 FEM 保持一致)
lx, ly = 1.0, 1.0
nx, ny = 64, 64
dt = 5e-5
T = 5e-3
if mode == 'train_init_steps':
    T = dt * 10
num_steps = int(T / dt)

# Cahn-Hilliard 参数
# M = 1.0
epsilon = 0.01
lambda_param = epsilon**2

# 谱方法网格设置
# 注意：numpy fftfreq 对应的频率顺序
kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx/nx)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly/ny)
KX, KY = np.meshgrid(kx, ky, indexing='xy') # indexing='xy' 对应 (ny, nx) 形状
K2 = KX**2 + KY**2
K4 = K2**2

# 初始条件生成器 (复刻 FEM 逻辑)
class InitialConditions:
    def __init__(self, seed, lx, ly, nx, ny, modes=100, kmax=15, amp=0.02):
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        np.random.seed(seed)
        
        self.num_modes = modes
        self.k_max = kmax
        self.noise_amp = amp
        
        self.kx_modes = np.random.randint(-self.k_max, self.k_max + 1, self.num_modes)
        self.ky_modes = np.random.randint(-self.k_max, self.k_max + 1, self.num_modes)
        self.phases = np.random.rand(self.num_modes) * 2 * np.pi
        self.scale = self.noise_amp / np.sqrt(self.num_modes)
        
        # 生成网格坐标 (不包含右/上边界，因为是周期性的)
        x = np.linspace(0, lx, nx, endpoint=False)
        y = np.linspace(0, ly, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='xy')

    def generate(self):
        val = np.zeros((self.ny, self.nx))
        for i in range(self.num_modes):
            kx_m = self.kx_modes[i]
            ky_m = self.ky_modes[i]
            phase = self.phases[i]
            
            theta = 2 * np.pi * (kx_m * self.X / self.lx + ky_m * self.Y / self.ly) + phase
            val += np.cos(theta)
        return val * self.scale

# 种子设置
if mode == 'train_valid':
    num_initials = 25
    initial_seeds = [100 + i for i in range(num_initials)]
    modes = [100] * num_initials
    kmaxs = [15] * num_initials
    np.random.seed(100)
    Ms = np.random.uniform(0.5, 1.5, num_initials)
    np.save(os.path.join(save_dir, 'M_values.npy'), Ms)


elif mode == 'test':
    num_initials = 5
    initial_seeds = [200 + i for i in range(num_initials)]
    # modes = [100] * num_initials
    # kmaxs = [15] * num_initials
    modes = [100] * num_initials
    kmaxs = [15] * num_initials
    Ms = [0.6, 0.8, 1.0, 1.2, 1.4]
    np.save(os.path.join(save_dir, 'M_values.npy'), np.array(Ms))
elif mode == 'train_init_steps':
    num_initials = 20
    initial_seeds = [300 + i for i in range(num_initials)]
    modes = [100] * num_initials
    kmaxs = [15] * num_initials

# 结果数组
# FEM 输出形状: (num_initials, steps, 1, ny, nx)
# 我们计算的是 (ny, nx)，不需要 padding
results_grid = np.zeros((num_initials, num_steps + 1, 1, ny, nx))
times = np.linspace(0, T, num_steps + 1)

print(f"开始批量模拟 (Spectral)... 模式: {mode}, 样本数: {num_initials}, 总步数: {num_steps}")

for i, seed in enumerate(initial_seeds):
    print(f"正在计算样本 {i+1}/{num_initials}, 种子: {seed}")
    M = Ms[i]
    
    # 1. 初始化 c
    init_gen = InitialConditions(seed, lx, ly, nx, ny, modes=modes[i], kmax=kmaxs[i], amp=0.01)
    c = init_gen.generate()
    
    # 保存初始状态 (t=0)
    # 不需要 padding
    results_grid[i, 0, 0] = c
    
    # 时间演化
    c_curr = c.copy()
    
    # 预计算系数 (Crank-Nicolson Fully Implicit Scheme)
    # 离散格式: (c_new - c_old)/dt = -M*K2 * 0.5*(f'(c_new) + f'(c_old)) - M*lambda*K4 * 0.5*(c_new + c_old)
    # 整理得: c_new * (1 + 0.5*dt*M*lambda*K4) = c_old * (1 - 0.5*dt*M*lambda*K4) - 0.5*dt*M*K2 * (f'(c_new) + f'(c_old))
    
    lhs_coef = 1.0 + 0.5 * dt * M * lambda_param * K4
    rhs_linear_coef = 1.0 - 0.5 * dt * M * lambda_param * K4
    nonlinear_factor = 0.5 * dt * M * K2
    
    for step in range(num_steps):
        c_hat_old = np.fft.fft2(c_curr)
        
        # 计算旧时刻的非线性项 f'(c_old) = c_old^3 - c_old
        f_prime_old = c_curr**3 - c_curr
        f_prime_hat_old = np.fft.fft2(f_prime_old)
        
        # RHS 中已知的部分 (Linear part + Explicit nonlinear part)
        rhs_known = rhs_linear_coef * c_hat_old - nonlinear_factor * f_prime_hat_old
        
        # Picard 迭代求解隐式非线性项
        # 初始猜测 c_new = c_old
        c_new = c_curr.copy()
        c_hat_new = c_hat_old.copy()
        
        max_iter = 100
        tol = 1e-9
        
        for k in range(max_iter):
            # 计算当前猜测的 f'(c_new)
            f_prime_new = c_new**3 - c_new
            f_prime_hat_new = np.fft.fft2(f_prime_new)
            
            # 更新 c_new_hat
            # lhs_coef * c_new_hat = rhs_known - nonlinear_factor * f_prime_hat_new
            c_hat_new_updated = (rhs_known - nonlinear_factor * f_prime_hat_new) / lhs_coef
            
            # 检查收敛性 (相对误差)
            diff = np.linalg.norm(c_hat_new_updated - c_hat_new) / (np.linalg.norm(c_hat_new) + 1e-12)
            
            c_hat_new = c_hat_new_updated
            c_new = np.fft.ifft2(c_hat_new).real
            
            if diff < tol:
                break
        
        # 更新 c
        c_curr = c_new
        
        # 保存
        results_grid[i, step + 1, 0] = c_curr
        
        if (step + 1) % 20 == 0:
            print(f"    完成步数 {step + 1}/{num_steps}")

# 保存结果
np.save(os.path.join(save_dir, 'solutions_grid.npy'), results_grid)
np.save(os.path.join(save_dir, 'initial_seeds.npy'), np.array(initial_seeds))
np.save(os.path.join(save_dir, 'times.npy'), times)


# 保存网格坐标 (与 FEM 格式一致)
# endpoint=False, 对应 64 个点
x_coords = np.linspace(0, lx, nx, endpoint=False)
y_coords = np.linspace(0, ly, ny, endpoint=False)
X_grid, Y_grid = np.meshgrid(x_coords, y_coords, indexing='xy')
grid_points = np.stack([X_grid, Y_grid], axis=-1)
np.save(os.path.join(save_dir, 'mesh_grid_coords.npy'), grid_points)

print(f"\n模拟完成! 结果已保存至 {save_dir}")
print(f"Grid shape: {results_grid.shape}")