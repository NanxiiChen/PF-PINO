from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    DEBUG_MODE = True

    # Model architecture settings
    model_type = "fno"  # Options: 'fno', 'fcn', 'unet'
    inception = True  # Use inception-style bypass blocks if True

    in_channels = 5 # phi, c, meshx, meshy, dt
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
    physical_residual = False

    save_dir = f"./phase_separation/runs/FNO-PI/" \
        if physical_residual else f"./phase_separation/runs/FNO/"
    if DEBUG_MODE:
        save_dir = save_dir[:-1] + "_DEBUG/"
    data_dir = "./phase_separation/data/train_valid/"
    test_data_dir = "./phase_separation/data/test/"

    Lc = 1.0 # xc = x / Lc
    Tc = 1e-4 # tc = t / Tc

    AC_PRE_SCALE = 1e0
    CH_PRE_SCALE = 1e5


    M = 1.0            # 迁移率
    epsilon = 0.01     # 界面厚度参数
    lambda_param = epsilon**2


