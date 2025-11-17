import os
import math
import fenics as fn
import numpy as np

mode = 'train_valid'
# mode = 'test' 

save_dir = './corrosion2d/data/train_valid' if mode == 'train_valid' else './corrosion2d/data/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
    
L1 = [-50e-6, 50e-6]
L2 = [0, 50e-6]
n = (100, 50)
mesh = fn.RectangleMesh(fn.Point(L1[0], L2[0]), 
                        fn.Point(L1[1], L2[1]), n[0], n[1])

Fun = fn.FiniteElement('P', mesh.ufl_cell(), 1)
pcFun = fn.FunctionSpace(mesh, Fun*Fun)


# solution variables, funtion and test functions
pc_sol = fn.Function(pcFun)
p_sol, c_sol = fn.split(pc_sol)
tpc = fn.TrialFunction(pcFun)
tc, tp = fn.split(tpc)
dpc = fn.TestFunction(pcFun)
dc, dp = fn.split(dpc)
# auxiliary variables for saving previous solutions
pc_t = fn.Function(pcFun)
p_t, c_t = fn.split(pc_t)

# Introduce manually the material parameters
alphap = 9.62e-5/4
omegap = 1.663e7
DD = 8.5e-10
AA = 5.35e7
Lp = 1.0e-10
cse = 1.
cle = 5100/1.43e5

def make_random_profile(xmin, xmax, n_modes=6, amp_scale=0.25, seed=None):
    rng = np.random.default_rng(seed)
    L = xmax - xmin
    ks = np.arange(1, n_modes+1)
    amps = rng.normal(scale=amp_scale/np.sqrt(ks), size=ks.shape)
    phases = rng.uniform(0, 2*np.pi, size=ks.shape)
    def profile(x):
        # x can be array or scalar
        arg = 2.0 * np.pi * (np.asarray(x) - xmin) / L
        sum_val = np.zeros_like(arg, dtype=float)
        for k, a, phi in zip(ks, amps, phases):
            sum_val += a * np.cos(k * arg + phi)
        base = L2[1] / 3 * 2
        return np.clip(base + sum_val, *L2)
    return profile


class InitialConditions(fn.UserExpression):
    def __init__(self, profile_callable, **kwargs):
        super().__init__(**kwargs)
        self.profile = profile_callable
        
    def eval_cell(self, values, x, ufc_cell):
        d = x[1] - self.profile(x[0])
        values[0] = (1 - math.tanh(math.sqrt(omegap) /
                                   math.sqrt(2 * alphap) * d)) / 2
        hp0 =  -2*values[0]**3 + 3*values[0]**2
        values[1] = hp0*cse + (1-hp0)*0.0
        
        
    def value_shape(self):
        return (2,)
    
    
def top_boundary(x, on_boundary):
    return on_boundary and fn.near(x[1], L2[1])

def bottom_boundary(x, on_boundary):
    return on_boundary and fn.near(x[1], L2[0])

bc_pc = [fn.DirichletBC(pcFun.sub(0), fn.Constant(0.0), top_boundary),
          fn.DirichletBC(pcFun.sub(1), fn.Constant(0.0), top_boundary),
          fn.DirichletBC(pcFun.sub(0), fn.Constant(1.0), bottom_boundary),
          fn.DirichletBC(pcFun.sub(1), fn.Constant(1.0), bottom_boundary)]
    
    

total_time = 2e4
num_steps = 100
dt = total_time / num_steps
times = np.linspace(0, total_time, num_steps + 1)

# build coordinate arrays for both FEM dofs and a regular rectangular grid
# preserve original dof coordinates (as before) but also create and save a regular grid
dof_points = pcFun.tabulate_dof_coordinates().reshape((-1, mesh.geometry().dim()))
dof_points_p = dof_points[::2]
# regular grid matching mesh resolution (nodes: nx+1, ny+1)
nx, ny = n
x_coords = np.linspace(L1[0], L1[1], nx + 1)
y_coords = np.linspace(L2[0], L2[1], ny + 1)
X, Y = np.meshgrid(x_coords, y_coords, indexing='xy')
grid_points = np.vstack([X.ravel(), Y.ravel()]).T
# save both coordinate sets
np.save(f'./{save_dir}/mesh_dof_coords.npy', dof_points_p)
np.save(f'./{save_dir}/mesh_grid_coords.npy', grid_points.reshape((ny + 1, nx + 1, 2)))

num_initials = 20 if mode == 'train_valid' else 5
initial_seeds = [100 + i for i in range(num_initials)] \
    if mode == 'train_valid' \
    else [200 + i for i in range(num_initials)]

# keep original flat-point results and add a grid-shaped results buffer (channels, y, x)
num_points = dof_points_p.shape[0]
num_grid_points = grid_points.shape[0]
results = np.zeros((num_initials, num_steps + 1, 2, num_points))
results_grid = np.zeros((num_initials, num_steps + 1, 2, ny + 1, nx + 1))


dx = fn.dx()
DT = fn.Constant(dt)
E_pc_template = (c_sol - c_t)/DT*dc*dx +\
    -fn.inner(-DD*fn.grad(c_sol) + DD*(cse-cle)*fn.grad(-2*p_sol**3 + 3*p_sol**2), fn.grad(dc))*dx +\
    (p_sol - p_t)/DT/Lp*dp*dx +\
    -2*AA*(c_sol - (-2*p_sol**3 + 3*p_sol**2)*(cse-cle) - cle)*(cse-cle)*(-6*p_sol**2 + 6*p_sol)*dp*dx +\
    omegap*(4*p_sol**3 - 6*p_sol**2 + 2*p_sol)*dp*dx +\
    fn.inner(alphap*fn.grad(p_sol), fn.grad(dp))*dx

Jpc_template = fn.derivative(E_pc_template, pc_sol, tpc)


for i, seed in enumerate(initial_seeds):
    profile = make_random_profile(L1[0], L1[1], n_modes=3, amp_scale=2.0e-6, seed=seed)
    init_expr = InitialConditions(profile, degree=2)

    pc_sol.interpolate(init_expr)
    pc_t.interpolate(init_expr)

    pc_vec = pc_sol.vector().get_local()
    p_vals = pc_vec[0::2]
    c_vals = pc_vec[1::2]
    results[i, 0, 0, :] = p_vals
    results[i, 0, 1, :] = c_vals

    # sample initial condition on regular grid and store in grid-shaped buffer
    sampled_init = np.array([tuple(pc_sol((float(px), float(py)))) for px, py in grid_points])
    # sampled_init shape -> (num_grid, 2) ; reshape to (2, ny+1, nx+1)
    results_grid[i, 0] = sampled_init.reshape((ny + 1, nx + 1, 2)).transpose(2, 0, 1)

    problem = fn.NonlinearVariationalProblem(E_pc_template, pc_sol, bc_pc, J=Jpc_template)
    solver_pc = fn.NonlinearVariationalSolver(problem)
    solver_pc.parameters['newton_solver']['absolute_tolerance'] = 1E-8
    solver_pc.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver_pc.parameters['newton_solver']["convergence_criterion"] = "incremental"
    solver_pc.parameters['newton_solver']["relative_tolerance"] = 1e-6
    solver_pc.parameters['newton_solver']["maximum_iterations"] = 10

    print(f"Initial {i+1}/{num_initials}, seed={seed}, Lp={Lp}")
    for step in range(num_steps):
        try:
            solver_pc.solve()
        except Exception as e:
            print("Solver failed at initial", i, "step", step, "error:", e)
            break

        pc_vec = pc_sol.vector().get_local()
        p_vals = pc_vec[0::2]
        c_vals = pc_vec[1::2]
        results[i, step + 1, 0, :] = p_vals
        results[i, step + 1, 1, :] = c_vals

        # --- 新增：在规则网格上采样 pc_sol 并保存（额外开销） ---
        sampled = np.array([tuple(pc_sol((float(px), float(py)))) for px, py in grid_points])
        # sampled.shape == (num_grid, 2)
        results_grid[i, step + 1] = sampled.reshape((ny + 1, nx + 1, 2)).transpose(2, 0, 1)
        # --- end sampled ---

        # 更新上一步解
        pc_t.vector()[:] = pc_sol.vector()

    print(f"Completed initial {i+1}/{num_initials}")

# 保存结果与元数据
np.save(f'{save_dir}/solutions_initials.npy', results)
np.save(f'{save_dir}/solutions_grid_initials.npy', results_grid)
np.save(f'{save_dir}/initial_seeds.npy', np.array(initial_seeds))
np.save(f'{save_dir}/times.npy', times)
print(f"Saved results to {save_dir}, shape: {results.shape}, grid-shape: {results_grid.shape}")