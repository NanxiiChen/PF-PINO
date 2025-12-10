import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split

import argparse

from .configs import load_configs
from .losses import Losses
from .model2d import get_model2d


def dataloader(
    key,
    dataset_x,
    dataset_y,
    batch_size,
    down_scale=1,
):
    n_samples = dataset_x.shape[0]
    n_batches = n_samples // batch_size

    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        indices = permutation[start:end]
        yield (
            dataset_x[indices, ..., ::down_scale, ::down_scale],
            dataset_y[indices, ..., ::down_scale, ::down_scale],
        )


@eqx.filter_jit
def train_step(model, loss_fn, state, optimizer, xs, ys, **kwargs):
    (loss, _), grad = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, xs, ys, **kwargs)
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss

@eqx.filter_jit
def train_step_pi(model, loss_fn, state, optimizer, 
                  xs, ys, dx, dy, dt, configs, **kwargs):
    (weighted_loss, (loss_components, weight_components, aux_vars)), grad = loss_fn(
        model, xs, ys, dx, dy, dt, configs, **kwargs
    )
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, weighted_loss, loss_components, weight_components, aux_vars

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--configs', type=str, default='train_debug', 
                            help='Configuration file for training')
    args = arg_parser.parse_args()
    configs = load_configs(args.configs).Configs()
    data = jnp.load(os.path.join(configs.data_dir, 
                                 "dataset_2d_complete.npz"))
    Xs = data["Xs"]  # (samples, 4, nx, ny)
    Ys = data["Ys"]  # (samples, 2, nx, ny)
    meshes = data["meshes"]  # (nx, ny, 2), normalized
    meshes = meshes[..., ::configs.down_scale, ::configs.down_scale]
    times = data["times"]  
    dt = times[1] - times[0]  # normalized
    dx = meshes[0, 0, 1] - meshes[0, 0, 0]  # normalized
    dy = meshes[1, 1, 0] - meshes[1, 0, 0]  # normalized
    print(f"dt: {dt}, dx: {dx}, dy: {dy}")
    train_x_full, valid_x_full, train_y_full, valid_y_full = train_test_split(
        Xs, Ys, test_size=0.25, random_state=0
    )
    jnp.savez(os.path.join(configs.data_dir, "dataset_split.npz"),
             train_x=train_x_full,
             train_y=train_y_full,
             valid_x=valid_x_full,
             valid_y=valid_y_full)
    print(f"Train Dataset shape: x {train_x_full.shape}, y {train_y_full.shape}")
    print(f"Valid Dataset shape: x {valid_x_full.shape}, y {valid_y_full.shape}")
    test_solutions = jnp.load(os.path.join(configs.test_data_dir, "solutions_grid.npy"))
    test_meshes = jnp.load(os.path.join(configs.test_data_dir, "mesh_grid_coords.npy"))
    test_meshes = jnp.transpose(test_meshes, (2, 0, 1))  # (samples, 2, nx, ny)
    test_times = jnp.load(os.path.join(configs.test_data_dir, "times.npy"))
    print(f"Test Dataset shape: solutions {test_solutions.shape}, meshes {test_meshes.shape},")
    test_times = test_times / configs.Tc
    test_meshes = test_meshes / configs.Lc
    test_dt = test_times[1] - test_times[0]
    x_test = test_solutions[:, 0, :, :, :] # (samples, channel, nx, ny)
    steps = test_solutions.shape[1]-1
    y_test = test_solutions[:, 1:, :, :, :] # (samples, channel, nx, ny)

    model_kwargs = {
        'modes_x': configs.modes_x,
        'modes_y': configs.modes_y,
        'width': configs.width,
        'depth': configs.depth,
        'activation': getattr(jax.nn, configs.activation),
        'key': jax.random.PRNGKey(0),
    }

    model = get_model2d(
        configs.model_type,
        in_channels=configs.in_channels,
        out_channels=configs.out_channels,
        **model_kwargs
    )

    losses = Losses()
    loss_fn = losses.pi_loss if configs.physical_residual else losses.mse_loss

    scheduler = optax.exponential_decay(
        init_value=configs.learning_rate,
        transition_steps=configs.decay_every,
        decay_rate=configs.decay_rate,
        end_value=configs.end_value,
    )
    optimizer = optax.adam(scheduler)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    batch_size = configs.batch_size
    train_loss_history = []
    valid_loss_history = []
    shuffle_key = jax.random.PRNGKey(0)
    savedir = configs.save_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    with open(os.path.join(savedir, "logs.csv"), "w") as f:
        f.write("Epoch,TrainLoss,ValidLoss,CHLoss,PotLoss\n")

    with open(os.path.join(savedir, "test_logs.csv"), "w") as f:
        f.write("Epoch,TestMSE\n")
    
    for epoch in range(configs.epochs):
        pde_name = "ch" if epoch % 50 < 25 else "pot"
        # pde_name = "both"
        shuffle_key, train_key, valid_key = jax.random.split(shuffle_key, 3)
        train_loader = dataloader(train_key, train_x_full, train_y_full, batch_size=batch_size, down_scale=configs.down_scale)
        valid_loader = dataloader(valid_key, valid_x_full, valid_y_full, batch_size=batch_size, down_scale=1)
        train_loss_epoch = 0.0
        val_loss_epoch = 0.0
        ch_loss_epoch = 0.0
        pot_loss_epoch = 0.0
        for train_batch_x, train_batch_y in train_loader:
            if configs.physical_residual:
                model, opt_state, weighted_loss, loss_components, weight_components, aux_vars = train_step_pi(
                    model, loss_fn,
                    opt_state, optimizer, 
                    train_batch_x, train_batch_y,
                    dx=dx, dy=dy,
                    dt=dt, configs=configs,
                    pde_name=pde_name,
                )
                train_loss_epoch += loss_components[0].item() * train_batch_x.shape[0]
                if pde_name == "both":
                    ch_loss_epoch += loss_components[1].item() * train_batch_x.shape[0]
                    pot_loss_epoch += loss_components[2].item() * train_batch_x.shape[0]
                elif pde_name == "ch":
                    ch_loss_epoch += loss_components[1].item() * train_batch_x.shape[0]
                elif pde_name == "pot":
                    pot_loss_epoch += loss_components[1].item() * train_batch_x.shape[0]
                else:
                    raise ValueError(f"Unknown pde_name: {pde_name}")


            else:
                model, opt_state, loss = train_step(
                    model, loss_fn,
                    opt_state, optimizer,
                    train_batch_x, train_batch_y,
                )
                train_loss_epoch += loss.item() * train_batch_x.shape[0]
        # Samples may be dropped to ensure all batches have equal size
        # So the actual number of samples used is (num_batches * batch_size)
        train_loss_epoch /= (train_x_full.shape[0] // batch_size * batch_size)
        train_loss_history.append(train_loss_epoch)
        if configs.physical_residual:
            ch_loss_epoch /= (train_x_full.shape[0] // batch_size * batch_size)
            pot_loss_epoch /= (train_x_full.shape[0] // batch_size * batch_size)
        
        for val_batch_x, val_batch_y in valid_loader:

            val_loss, _ = losses.mse_loss_weighted(model, 
                                          xs=val_batch_x,
                                          ys=val_batch_y,)
            val_loss_epoch += val_loss.item() * val_batch_x.shape[0]
        val_loss_epoch /= (valid_x_full.shape[0] // batch_size * batch_size)
        valid_loss_history.append(val_loss_epoch)
        
        with open(os.path.join(savedir, "logs.csv"), "a") as f:
            if pde_name == "both":
                f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch},{ch_loss_epoch},{pot_loss_epoch}\n")
            elif pde_name == "ch":
                f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch},{ch_loss_epoch},{jnp.nan}\n")
            elif pde_name == "pot":
                f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch},{jnp.nan},{pot_loss_epoch}\n")
            else:
                raise ValueError(f"Unknown pde_name: {pde_name}")
    

        if epoch % configs.save_every == 0 or epoch == configs.epochs - 1:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss_epoch:.3e}, "
                f"Valid Loss = {val_loss_epoch:.3e}"
            )
            

        if epoch % configs.test_every == 0 or epoch == configs.epochs - 1:
            auto_reg_fn = partial(
                model.auto_reg,
                meshes=test_meshes,
                steps=steps,
            )
            y_test_pred = jax.vmap(auto_reg_fn, in_axes=(0))(x_test)
            test_mse = jnp.mean((y_test_pred - y_test) ** 2)
            print(f"Test MSE at epoch {epoch}: {test_mse:.3e}")
            with open(os.path.join(savedir, "test_logs.csv"), "a") as f:
                f.write(f"{epoch},{test_mse:.6e}\n")

            eqx.tree_serialise_leaves(
                os.path.join(savedir, f"epoch_{epoch}.eqx"),
                model)
            

if __name__ == "__main__":
    main()