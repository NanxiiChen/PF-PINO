from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import scipy
from sklearn.model_selection import train_test_split

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
def vmapped_forward(model:FNO1d, x:jnp.ndarray, **kwargs) -> jnp.ndarray:
    return jax.vmap(model)(x)


@eqx.filter_jit
def loss_fn(model, x, y, **kwargs):
    y_pred = vmapped_forward(model, x, **kwargs)
    loss = jnp.mean(jnp.square(y_pred - y))
    return loss


@eqx.filter_jit
def train_step(model, state, optimizer, x, y, **kwargs):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y, **kwargs)
    updates, new_state = optimizer.update(grad, state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss

def main():
    # data = scipy.io.loadmat("burgers_data_R10.mat")
    # a, u = data["a"], data["u"]
    # a = data["a"][:, None, :]
    # u = data["u"][:, None, :]
    data = jnp.load("./dataset_1d_complete.npz")
    Xs, Ys = data["Xs"], data["Ys"]
    meshes = data["meshes"]
    times = data["times"]
    dt = times[1] - times[0]
    steps = Ys.shape[1]
    train_x_full, valid_x_full, train_y_full, valid_y_full = train_test_split(
        Xs, Ys, test_size=0.2, random_state=0
    )
    jnp.savez("./dataset_split.npz",
             train_x=train_x_full,
             train_y=train_y_full,
             valid_x=valid_x_full,
             valid_y=valid_y_full)
    
    fno = FNO1d(
        5, 2, 32, 128, 4,
        activation=jax.nn.gelu,
        key=jax.random.PRNGKey(0),
    )

    schedualer = optax.exponential_decay(
        init_value=0.001,
        transition_steps=100,
        decay_rate=0.95,
        end_value=5e-5,
    )
    optimizer = optax.adam(schedualer)
    opt_state = optimizer.init(eqx.filter(fno, eqx.is_array))


    batch_size = 64
    train_loss_history = []
    valid_loss_history = []
    shuffle_key = jax.random.PRNGKey(0)
    
    with open("./logs/loss_log.csv", "w") as f:
        f.write("Epoch,TrainLoss,ValidLoss\n")
    
        
    
    for epoch in range(2000):
        shuffle_key, train_key, subkey = jax.random.split(shuffle_key, 3)
        train_loader = dataloader(train_key, train_x_full, train_y_full, batch_size=batch_size)
        valid_loader = dataloader(subkey, valid_x_full, valid_y_full, batch_size=batch_size)
        train_loss_epoch = 0.0
        val_loss_epoch = 0.0
        for train_batch_x, train_batch_y in train_loader:
            fno, opt_state, loss = train_step(
                fno, opt_state, optimizer, 
                train_batch_x, train_batch_y,
                meshes=meshes, steps=steps, dt=dt
            )
            train_loss_epoch += loss.item() * train_batch_x.shape[0]
        train_loss_epoch /= train_x_full.shape[0]
        train_loss_history.append(train_loss_epoch)
        
        for val_batch_x, val_batch_y in valid_loader:
            val_loss = loss_fn(fno, val_batch_x, val_batch_y, 
                               meshes=meshes, steps=steps, dt=dt)
            val_loss_epoch += val_loss.item() * val_batch_x.shape[0]
        val_loss_epoch /= valid_x_full.shape[0]
        valid_loss_history.append(val_loss_epoch)
        
        with open("./logs/loss_log.csv", "a") as f:
            f.write(f"{epoch},{train_loss_epoch},{val_loss_epoch}\n")
            
        if epoch % 100 == 0:
            print(
                f"Epoch {epoch}: "
                f"Train Loss = {train_loss_epoch:.3e}, "
                f"Valid Loss = {val_loss_epoch:.3e}"
            )
            
            eqx.tree_serialise_leaves(
                f"./weights/fno_epoch_{epoch}.eqx", fno
            )
            
if __name__ == "__main__":
    main()
