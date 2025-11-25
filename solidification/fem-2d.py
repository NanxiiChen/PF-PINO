from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time
import os

set_log_level(LogLevel.ERROR)

# ==================== 参数设置（Example 5.3）====================
# 模型参数
rho_val = 1e3        # ϱ(φ) - 迁移率倒数（这里取常数）
eps = 0.015          # ε - 界面宽度
sigma = 0.1          # σ - 各向异性强度
lam = 4e2            # λ - 动力学系数
D_val = 2.5e-3       # D - 热扩散系数
K_val = 1.2          # K - 潜热系数

# 数值参数
Nx, Ny = 128, 128    # 网格分辨率
T_final = 10        # 终止时间
dt = 0.005           # 时间步长（直接法需要小步长）
num_steps = int(T_final / dt)
save_interval = 10

# 初始条件参数
r0 = 0.05
x0, y0 = 0.0, 0.0
eps0 = eps

# ==================== 网格和函数空间 ====================
mesh = RectangleMesh(Point(-1, -1), Point(1, 1), Nx, Ny, 'crossed')
V = FunctionSpace(mesh, 'P', 1)

print(f"自由度: {V.dim()}")
print(f"时间步长: {dt}, 总步数: {num_steps}")

# ==================== 函数定义 ====================
# 当前时刻
phi = Function(V)
T_var = Function(V)

# [新增] 显式重命名，防止出现 f_11, f_5 这种名字
phi.rename("phi", "phase field")
T_var.rename("T", "temperature")

# 上一时刻
phi_n = Function(V)
T_n = Function(V)


# ==================== 初始条件 ====================
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

phi_n.interpolate(PhiInitial(degree=2))
T_n.interpolate(TInitial(degree=2))

# ==================== 模型函数 ====================
def F_potential(phi):
    """F(φ) = (φ² - 1)² / 4"""
    return 0.25 * (phi**2 - 1)**2

def F_prime(phi):
    """F'(φ) = φ³ - φ"""
    return phi**3 - phi

def h_func(phi):
    """h(φ) = φ⁵/5 - 2φ³/3 + φ"""
    return phi**5 / 5.0 - 2.0 * phi**3 / 3.0 + phi

def h_prime(phi):
    """h'(φ) = φ⁴ - 2φ² + 1"""
    return phi**4 - 2*phi**2 + 1

def kappa(grad_phi):
    """各向异性系数 κ(∇φ) = 1 + σ·cos(4θ)"""
    eps_reg = 1e-12
    phi_x, phi_y = grad_phi[0], grad_phi[1]
    # cos(2θ) = (φ_x² - φ_y²) / |∇φ|²
    # sin(2θ) = 2φ_xφ_y / |∇φ|²
    # cos(4θ) = cos²(2θ) - sin²(2θ)
    norm_sq = phi_x**2 + phi_y**2 + eps_reg
    cos2theta = (phi_x**2 - phi_y**2) / norm_sq
    sin2theta = 2 * phi_x * phi_y / norm_sq
    cos4theta = cos2theta**2 - sin2theta**2
    return 1 + sigma * cos4theta

def compute_H(grad_phi):
    """
    H(∇φ) = δℒ/δφ，其中 ℒ = ∫ κ(∇φ) dΩ
    对于 m=4:
    H = 4σ·4/|∇φ|⁶ · (φ_x(φ_x²φ_y² - φ_y⁴), φ_y(φ_y²φ_x² - φ_x⁴))
    """
    phi_x, phi_y = grad_phi[0], grad_phi[1]
    norm6 = (phi_x**2 + phi_y**2)**3 + 1e-18
    coef = 16.0 * sigma / norm6
    Hx = coef * phi_x * (phi_x**2 * phi_y**2 - phi_y**4)
    Hy = coef * phi_y * (phi_y**2 * phi_x**2 - phi_x**4)
    return as_vector((Hx, Hy))

# ==================== 求解器：直接求解方程(2.1)和(2.2) ====================

def solve_direct():
    """
    直接求解原始方程：
    方程(2.1): ϱ(φ)·∂φ/∂t = -δE/δφ - (λ/ε)h'(φ)T
    方程(2.2): ∂T/∂t = ∇·(D∇T) + K·h'(φ)·∂φ/∂t
    
    使用隐式时间离散
    """
    
    # 测试函数
    v_phi = TestFunction(V)
    v_T = TestFunction(V)
    
    # 试探函数
    phi_trial = TrialFunction(V)
    T_trial = TrialFunction(V)
    
    # ==================== 方程(2.1)的弱形式 ====================
    # ϱ(φⁿ)·(φⁿ⁺¹ - φⁿ)/dt = -δE/δφ|_{φⁿ⁺¹} - (λ/ε)h'(φⁿ)Tⁿ
    
    # δE/δφ = -∇·[κ²(∇φ)∇φ + κ(∇φ)|∇φ|²H(φ)] + f(φ)/ε²
    
    # 使用半隐式：φⁿ⁺¹的线性项隐式，非线性项显式
    # 简化：用φⁿ计算κ, H, f等非线性项
    
    grad_phi_n = grad(phi_n)
    kappa_n = kappa(grad_phi_n)
    H_n = compute_H(grad_phi_n)
    
    # 弱形式（分部积分）：
    # ∫ ϱ(φⁿ)·(φⁿ⁺¹ - φⁿ)/dt·v dΩ
    # + ∫ [κ²(∇φⁿ)∇φⁿ⁺¹ + κ(∇φⁿ)|∇φⁿ|²H(φⁿ)]·∇v dΩ
    # + ∫ f(φⁿ)/ε²·v dΩ
    # + ∫ (λ/ε)h'(φⁿ)Tⁿ·v dΩ = 0
    
    # 为了稳定性，对梯度项用半隐式
    df_dphi_semi = (phi_n**2 - 1) * phi_trial
    F_phi = (
        rho_val * (phi_trial - phi_n) / dt * v_phi * dx
        + kappa_n**2 * dot(grad(phi_trial), grad(v_phi)) * dx
        + kappa_n * dot(grad_phi_n, grad_phi_n) * dot(H_n, grad(v_phi)) * dx
        + df_dphi_semi / eps**2 * v_phi * dx
        + (lam / eps) * h_prime(phi_n) * T_n * v_phi * dx
    )
    
    # 提取双线性和线性部分
    a_phi = lhs(F_phi)
    L_phi = rhs(F_phi)
    
    # 求解 φⁿ⁺¹
    solve(a_phi == L_phi, phi, [])
    
    
    # ==================== 方程(2.2)的弱形式 ====================
    # ∂T/∂t = D·ΔT + K·h'(φ)·∂φ/∂t
    # (Tⁿ⁺¹ - Tⁿ)/dt = D·ΔTⁿ⁺¹ + K·h'(φⁿ)·(φⁿ⁺¹ - φⁿ)/dt
    
    # 弱形式：
    # ∫ (Tⁿ⁺¹ - Tⁿ)/dt·v dΩ
    # + ∫ D·∇Tⁿ⁺¹·∇v dΩ
    # - ∫ K·h'(φⁿ)·(φⁿ⁺¹ - φⁿ)/dt·v dΩ = 0
    
    F_T = (
        (T_trial - T_n) / dt * v_T * dx
        + D_val * dot(grad(T_trial), grad(v_T)) * dx
        - K_val * h_prime(phi_n) * (phi - phi_n) / dt * v_T * dx
    )
    
    a_T = lhs(F_T)
    L_T = rhs(F_T)
    
    # 求解 Tⁿ⁺¹
    solve(a_T == L_T, T_var, [])
    
    return phi, T_var


# ==================== 计算能量 ====================
def compute_energy():
    """计算总能量 E(φ,T) = ∫[½κ²|∇φ|² + F(φ)/ε² + λT²/(2εK)] dΩ"""
    grad_phi_n = grad(phi_n)
    kappa_n = kappa(grad_phi_n)
    
    energy = assemble(
        (0.5 * kappa_n**2 * dot(grad_phi_n, grad_phi_n) 
         + F_potential(phi_n) / eps**2
         + lam * T_n**2 / (2 * eps * K_val)) * dx
    )
    return energy


# ==================== 时间步进循环 ====================
print("="*70)
print("开始直接求解方程(2.1)和(2.2)...")
print("="*70)

# 创建输出文件夹
os.makedirs("results", exist_ok=True)

phi_file = File("results/phi.pvd")
T_file = File("results/temperature.pvd")

t = 0.0
start_time = time.time()

# [修改] 保存初始状态时，先将值赋给主变量，保证变量名一致
phi.assign(phi_n)
T_var.assign(T_n)

phi_file << (phi, t)      # 原来是 phi_n
T_file << (T_var, t)      # 原来是 T_n

energies = [compute_energy()]
times = [0.0]

for n in range(num_steps):
    t += dt
    
    # 求解当前步
    phi, T_var = solve_direct()
    
    # 计算能量
    energy = compute_energy()
    energies.append(energy)
    times.append(t)
    
    # 输出信息
    if n % 10 == 0:
        elapsed = time.time() - start_time
        phi_min, phi_max = phi.vector().min(), phi.vector().max()
        T_min, T_max = T_var.vector().min(), T_var.vector().max()
        print(f"步数 {n:4d}/{num_steps}, t={t:.4f}, "
              f"E={energy:.6e}, "
              f"φ∈[{phi_min:.3f},{phi_max:.3f}], "
              f"T∈[{T_min:.3f},{T_max:.3f}], "
              f"耗时={elapsed:.1f}s")
    
    # 更新历史值
    phi_n.assign(phi)
    T_n.assign(T_var)
    
    # 保存结果
    if n % save_interval == 0:
        phi_file << (phi, t)
        T_file << (T_var, t)

print("="*70)
print("计算完成！")
print(f"总耗时: {time.time() - start_time:.2f}s")

# ==================== 绘制能量演化 ====================
plt.figure(figsize=(10, 6))
plt.plot(times, energies, 'b-', linewidth=2)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Energy E(φ,T)', fontsize=14)
plt.title('Energy Dissipation', fontsize=16)
plt.grid(True, alpha=0.3)
plt.savefig('results/energy.png', dpi=150, bbox_inches='tight')
print("能量曲线已保存到 results/energy.png")

# 检查能量是否单调递减
energy_diff = np.diff(energies)
if np.all(energy_diff <= 1e-10):
    print("✓ 能量单调递减（数值稳定）")
else:
    n_increase = np.sum(energy_diff > 1e-10)
    print(f"✗ 能量有 {n_increase} 次上升（可能需要减小时间步长）")

plt.show()