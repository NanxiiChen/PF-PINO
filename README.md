# Corrosion modelling 1D using FNO

## Dataset Generation
We parameterize the interfacial kinetics $L_p$.

For training and validation, we use the following $L_p$ values:

```python
Lp_values = [1.0e-9, 5.0e-9, 1.0e-8, 5.0e-8, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0e0]
```
with 100 time steps and 100 elements.

FEM solution is obtained using FEniCS and saved in `results-1d`, reulting in the total solutions of [12, 101, 2, 101] shape, i.e., [number of Lp values, time steps, channels, spatial points].

We train a one-step FNO model to learn:
$$
\phi(t+dt) = \mathcal{F}(phi(t), L_p)\\
c(t+dt) = \mathcal{F}(c(t), L_p)
$$
The `Xs` and `Ys` arrays are constructed accordingly, with:
- `Xs` shape: [12*100, 5, 101]
- `Ys` shape: [12*100, 2, 101]

The 5 channels in `Xs` correspond to:
1. $\phi$ at time t
2. $c$ at time t
3. $L_p$ (constant over space for each sample)
4. time t (constant over space for each sample)
5. spatial coordinate x (same vector for all samples)

For testing, we use a different set of $L_p$ values:
```python
Lp_values = [2.5e-9, 2.5e-8, 2.5e-7, 2.5e-6, 5.0e-3, 5.0e-1]
```
with the same time steps and elements. The FEM solutions are saved in `results-1d-test`.

Instead of predicting one-step ahead, we recursively apply auto-regressive predictions over the entire time sequence for testing.
```python
@eqx.filter_jit
def auto_reg(self, u0, Lp, meshes, dt, steps):
    # meshes: [S,]
    meshes = meshes[None, :]  # [1, S]
    preds = []
    # vmap outside the function, 
    # so u0 shape is [C, S] without B
    u = u0 
    for step in range(steps):
        tic = step * dt
        t_channel = jnp.full_like(meshes, tic)
        Lp_channel = jnp.full_like(meshes, Lp)
        inputs = jnp.concatenate([u, Lp_channel, 
                                    meshes, t_channel], 
                                    axis=0)  # [C+2, S]
        u = self.__call__(inputs)  # [C, S]
        preds.append(u)
    return jnp.stack(preds, axis=0)  # [T, C, S]
```
The test inputs have the shape:
- `u0` shape: [2, 101] (initial condition at t=0)
- `Lp` shape: scalar
- `meshes` shape: [101,]
- `dt`: scalar
- `steps`: integer (number of time steps to predict)

The experimental results show acceptable accuracy despite the risk of error accumulation in auto-regressive predictions.
Futher improvements can be made by exploring unrolling K steps during training and introducing physics-informed losses.