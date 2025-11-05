import os

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from sklearn.model_selection import train_test_split

from configs import Configs
from losses import Losses
from model import FNO1d


def dataloader(
    key,
    dataset_x,
    dataset_y,
    batch_size,
    down_scale=1
):
    n_samples = dataset_x.shape[0]

    n_batches = int(jnp.ceil(n_samples / batch_size))

    permutation = jax.random.permutation(key, n_samples)

    for batch_id in range(n_batches):
        start = batch_id * batch_size
        end = min((batch_id + 1) * batch_size, n_samples)

        batch_indices = permutation[start:end]

        yield dataset_x[batch_indices, ..., ::down_scale], dataset_y[batch_indices, ..., ::down_scale]

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
                  xs, ys, Lps, dx, dt, configs, **kwargs):
    (weighted_loss, (loss_components, weight_components, aux_vars)), grad = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model, xs, ys, Lps, dx, dt, configs, **kwargs)
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, weighted_loss, loss_components, weight_components, aux_vars



def main():
    # data = scipy.io.loadmat("burgers_data_R10.mat")
    # a, u = data["a"], data["u"]
    # a = data["a"][:, None, :]
    # u = data["u"][:, None, :]
    configs = Configs()
    data = jnp.load(os.path.join(configs.data_dir, "dataset_1d_complete.npz"))
    Xs, Ys = data["Xs"], data["Ys"]
    meshes = data["meshes"] # already normalized
    times = data["times"] # already normalized
    dt = times[1] - times[0]
    dx = meshes[1] - meshes[0]
    steps = Ys.shape[1]
    train_x_full, valid_x_full, train_y_full, valid_y_full = train_test_split(
        Xs, Ys, test_size=0.2, random_state=0
    )
    jnp.savez(os.path.join(configs.data_dir, "dataset_split.npz"),
             train_x=train_x_full,
             train_y=train_y_full,
             valid_x=valid_x_full,
             valid_y=valid_y_full)
    
    fno = FNO1d(
        configs.in_channels,
        configs.out_channels,
        configs.modes,
        configs.width,
        configs.depth,
        activation=getattr(jax.nn, configs.activation),
        key=jax.random.PRNGKey(0),
    )
    
    losses = Losses()
    loss_fn = losses.pi_loss

    schedualer = optax.exponential_decay(
        init_value=configs.learning_rate,
        transition_steps=configs.decay_every,
        decay_rate=configs.decay_rate,
        end_value=configs.end_value,
    )
    optimizer = optax.adam(schedualer)
    opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))


    batch_size = configs.batch_size
    train_loss_history = []
    valid_loss_history = []
    shuffle_key = jax.random.PRNGKey(0)
    savedir = configs.save_dir
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    with open(os.path.join(savedir, "logs.csv"), "w") as f:
        f.write("Epoch,TrainLoss,ValidLoss\n")
    
    for epoch in range(configs.epochs):
        shuffle_key, train_key, subkey = jax.random.split(shuffle_key, 3)
        train_loader = dataloader(train_key, train_x_full, train_y_full, batch_size=batch_size)
        valid_loader = dataloader(subkey, valid_x_full, valid_y_full, batch_size=batch_size)
        train_loss_epoch = 0.0
        val_loss_epoch = 0.0
        for train_batch_x, train_batch_y in train_loader:
            fno, opt_state, weighted_loss, loss_components, weight_components, aux_vars = train_step_pi(
                fno, loss_fn,
                opt_state, optimizer, 
                train_batch_x, train_batch_y,
                Lps=configs.Lp_from_Lpc(train_batch_x[:, 2, 0]),
                dx=dx, dt=dt, configs=configs
            )
            train_loss_epoch += loss_components[0].item() * train_batch_x.shape[0]
        train_loss_epoch /= train_x_full.shape[0]
        train_loss_history.append(train_loss_epoch)
        
        for val_batch_x, val_batch_y in valid_loader:
            val_loss, _ = losses.mse_loss(fno, val_batch_x, val_batch_y, meshes=meshes,)
            val_loss_epoch += val_loss.item() * val_batch_x.shape[0]
        val_loss_epoch /= valid_x_full.shape[0]
        valid_loss_history.append(val_loss_epoch)
        
        with open(os.path.join(savedir, "logs.csv"), "a") as f:
            f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch}\n")
            
        if epoch % configs.save_every == 0 or epoch == configs.epochs - 1:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss_epoch:.3e}, "
                f"Valid Loss = {val_loss_epoch:.3e}"
            )
            
            eqx.tree_serialise_leaves(
                os.path.join(savedir, f"epoch_{epoch}.eqx"),
                fno)
            
if __name__ == "__main__":
    main()
