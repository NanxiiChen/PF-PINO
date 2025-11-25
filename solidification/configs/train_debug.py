from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    DEBUG_MODE = True

    # Model architecture settings
    model_type = "fno"  # Options: 'fno', 'fcn', 'unet'

    in_channels = 6 # phi, c, K, meshx, meshy, dt
    out_channels = 2 # phi, c
    modes_x = 8
    modes_y = 8
    width = 64
    depth = 4
    activation = "relu"
    down_scale = 1


    learning_rate = 5e-4
    decay_every = 200
    decay_rate = 0.95
    end_value = 1e-5
    

    batch_size = 128
    epochs = 5000
    save_every = 100
    test_every = 500
    physical_residual = True

    save_dir = f"./solidification/runs/FNO-PI/"
    if DEBUG_MODE:
        save_dir = save_dir[:-1] + "_DEBUG/"
    data_dir = "./solidification/data/train_valid/"
    test_data_dir = "./solidification/data/test/"

    Lc = 1.0 # xc = x / Lc
    Tc = 1.0 # tc = t / Tc

    rho_val = 1e3        
    eps = 0.015          
    sigma = 0.1          
    lam = 4e2            
    D_val = 2.5e-3
    r0 = 0.05
