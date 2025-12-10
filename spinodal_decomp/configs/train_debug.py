from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    DEBUG_MODE = True

    # Model architecture settings
    model_type = "fno"  # Options: 'fno', 'fcn', 'unet'
    inception = True  # Use inception-style bypass blocks if True

    in_channels = 4 # c, mu, meshx, meshy
    out_channels = 2 # c, mu
    modes_x = 16
    modes_y = 16
    width = 64
    depth = 4
    activation = "gelu"
    down_scale = 1


    learning_rate = 5e-4
    decay_every = 50
    decay_rate = 0.95
    end_value = 1e-5
    

    batch_size = 128
    epochs = 5000
    save_every = 50
    test_every = 200
    physical_residual = True

    save_dir = f"./spinodal_decomp/runs/FNO-PI/" \
        if physical_residual else f"./spinodal_decomp/runs/FNO/"
    if DEBUG_MODE:
        save_dir = save_dir[:-1] + "_DEBUG/"
    data_dir = "./spinodal_decomp/data/train_valid/"
    test_data_dir = "./spinodal_decomp/data/test/"

    Lc = 1.0 # xc = x / Lc
    Tc = 1e-4 # tc = t / Tc

    CH_PRE_SCALE = 1e4
    POT_PRE_SCALE = 1e1

    M = 1.0            # 迁移率
    epsilon = 0.01     # 界面厚度参数
    lambda_param = epsilon**2


