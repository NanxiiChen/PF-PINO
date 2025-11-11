#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:56:03 2021
Pitting corrosion in FEniCs
@author: Sergio Lucarini
"""

import os
import math
import fenics as fn
import numpy as np

mode = 'train_valid'
# mode = 'test'

save_dir = './data/train_valid' if mode == 'train_valid' else './data/test'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Length and discretization of mesh
L = [-50e-6, 50e-6]
n = 100
# Create 1D mesh with n0 nodes
mesh = fn.IntervalMesh(n, L[0], L[1])
# export mesh point
# mesh_points = mesh.coordinates()
# print(mesh_points)
# np.save('mesh_points', mesh_points)
# mesh = fn.RectangleMesh(fn.Point(0, 0), fn.Point(
#     L[0], L[1]), n[0], n[1], "crossed")

# Define Space: Lagrange 1-linear 2-quadratic elements
Fun = fn.FiniteElement('CG', mesh.ufl_cell(), 1)
pcFun = fn.FunctionSpace(mesh, Fun * Fun)

# Get mesh points in the same order as the solution DOFs
mesh_points = pcFun.tabulate_dof_coordinates()[::2]  # Every 2nd coordinate (for p component)
print("Mesh points shape:", mesh_points.shape)
print("Mesh coordinate range:", mesh_points.min(), "to", mesh_points.max())
np.save(f'./{save_dir}/mesh_points.npy', mesh_points)

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
alphap = 9.62e-5
omegap = 1.663e7
DD = 8.5e-10
AA = 5.35e7

if mode == 'train_valid':
    # Lp parameter values for the train and validation sets
    Lp_values = [1.0e-9, 1.0e-8, 1.0e-7, 1.0e-5, 1.0e0]
elif mode == 'test':
    # Lower number of Lp values for testing
    Lp_values = [5.0e-9, 2.5e-8, 5.0e-7, 1.0e-6, 1.0e-3, 5.0e-1]
else:
    raise ValueError("Invalid mode. Choose 'train_valid' or 'test'.")
cse = 1.
cle = 5100/1.43e5

# Boundary conditions
def ConcentrationBoundary_left(x):
    return fn.near(x[0], L[0])

def ConcentrationBoundary_right(x):
    return fn.near(x[0], L[1])

bc_pc = [fn.DirichletBC(pcFun.sub(0), fn.Constant(1.0), ConcentrationBoundary_left),
         fn.DirichletBC(pcFun.sub(1), fn.Constant(1.0), ConcentrationBoundary_left),
         fn.DirichletBC(pcFun.sub(0), fn.Constant(0.0), ConcentrationBoundary_right),
         fn.DirichletBC(pcFun.sub(1), fn.Constant(0.0), ConcentrationBoundary_right)]

# Initial conditions as expression
class InitialConditions(fn.UserExpression):
    def eval_cell(self, values, x, ufc_cell):
        # values[0] = (1 - math.tanh(math.sqrt(omegap) /
        #    math.sqrt(2 * alphap) * x[0])) / 2
        # values[1] = (1 - math.tanh(math.sqrt(omegap) /
        #    math.sqrt(2 * alphap) * x[0])) / 2
        values[0] = (1 - math.tanh(math.sqrt(omegap) /
                                   math.sqrt(2 * alphap) * (x[0] - 35e-6))) / 2
        hp0 = -2*values[0]**3 + 3*values[0]**2
        values[1] = hp0*cse + (1-hp0)*0

    def value_shape(self):
        return (2,)

# Time parameters
total_time = 100.0
num_steps = 100
dt = total_time / num_steps
times = np.linspace(0, total_time, num_steps + 1)

# Initialize storage array: [num_Lp, num_time, num_variables, num_points]
num_points = mesh_points.shape[0]
results = np.zeros((len(Lp_values), num_steps + 1, 2, num_points))  # +1 for initial condition

# Loop over Lp values
for lp_idx, Lp in enumerate(Lp_values):
    print(f"Starting simulation with Lp = {Lp}")
    
    # Apply initial conditions
    pc_sol.interpolate(InitialConditions())
    pc_t.interpolate(InitialConditions())
    
    # Store initial condition
    pc_vec = pc_sol.vector().get_local()
    # Extract p and c values (interleaved: [p0, c0, p1, c1, ...])
    p_vals = pc_vec[0::2]  # Every even index (p values)
    c_vals = pc_vec[1::2]  # Every odd index (c values)
    results[lp_idx, 0, 0, :] = p_vals  # p values
    results[lp_idx, 0, 1, :] = c_vals  # c values
    
    # Weak Form with current Lp value
    dx = fn.dx()
    DT = fn.Constant(dt)
    E_pc = (c_sol-c_t)/DT*dc*dx +\
        -fn.inner(-DD*fn.grad(c_sol)+DD*(cse-cle)*fn.grad(-2*p_sol**3 + 3*p_sol**2), fn.grad(dc))*dx +\
        (p_sol-p_t)/DT/Lp*dp*dx +\
        -2*AA*(c_sol-(-2*p_sol**3 + 3*p_sol**2)*(cse-cle)-cle)*(cse-cle)*(-6*p_sol**2 + 6*p_sol)*dp*dx +\
        omegap*(4*p_sol**3-6*p_sol**2+2*p_sol)*dp*dx +\
        fn.inner(alphap*fn.grad(p_sol), fn.grad(dp))*dx

    # Automatically calculate jacobian
    Jpc = fn.derivative(E_pc, pc_sol, tpc)
    
    # Define the non linear problem and solver parameters
    p_pc = fn.NonlinearVariationalProblem(E_pc, pc_sol, bc_pc, J=Jpc)
    solver_pc = fn.NonlinearVariationalSolver(p_pc)
    solver_pc.parameters['newton_solver']['absolute_tolerance'] = 1E-8
    solver_pc.parameters['newton_solver']['linear_solver'] = 'mumps'
    solver_pc.parameters['newton_solver']["convergence_criterion"] = "incremental"
    solver_pc.parameters['newton_solver']["relative_tolerance"] = 1e-6
    solver_pc.parameters['newton_solver']["maximum_iterations"] = 10

    # Time stepping loop
    for step in range(num_steps):
        t = (step + 1) * dt
        print(f"Lp = {Lp}, Time step: {step+1}/{num_steps}, Time: {t:.3f}")

        # Newton solver
        try:
            info = solver_pc.solve()
        except:
            print("Solver not converged")
            break

        # Store solution
        pc_vec = pc_sol.vector().get_local()
        # Extract p and c values (interleaved: [p0, c0, p1, c1, ...])
        p_vals = pc_vec[0::2]  # Every even index (p values)
        c_vals = pc_vec[1::2]  # Every odd index (c values)
        results[lp_idx, step + 1, 0, :] = p_vals  # p values
        results[lp_idx, step + 1, 1, :] = c_vals  # c values
        
        # Update for next time step
        pc_t.vector()[:] = pc_sol.vector()

    print(f"Completed simulation for Lp = {Lp}")

# Save the complete results array
np.save(f'{save_dir}/solutions.npy', results)
np.save(f'{save_dir}/Lp_values.npy', Lp_values)
np.save(f'{save_dir}/times.npy', times)
print(f'Results saved with shape: {results.shape}')
print('Simulation complete')
