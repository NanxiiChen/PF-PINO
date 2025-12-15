from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    DEBUG_MODE = True

    # Model architecture settings
    model_type = "fno"  # Options: 'fno', 'fcn', 'unet'
    inception = True  # Use inception-style bypass blocks if True

    in_channels = 3 # c, meshx, meshy
    out_channels = 1 # c
    modes_x = 32
    modes_y = 32
    width = 64
    depth = 3
    activation = "gelu"
    down_scale = 1


    learning_rate = 5e-4
    decay_every = 200
    decay_rate = 0.95
    end_value = 5e-5
    

    batch_size = 64
    epochs = 5000
    save_every = 50
    test_every = 200
    physical_residual = True

    save_dir = f"/root/autodl-tmp/runs/spinodal_decomp/FNO-PI/"
    if DEBUG_MODE:
        save_dir = save_dir[:-1] + "_DEBUG/"
    data_dir = "/root/autodl-tmp/data/spinodal_decomp_spectra/train_valid/"
    test_data_dir = "/root/autodl-tmp/data/spinodal_decomp_spectra/test/"

    Lc = 1.0 # xc = x / Lc
    Tc = 1e-4 # tc = t / Tc

    CH_PRE_SCALE = 1e0
    POT_PRE_SCALE = 1e0

    M = 1.0            # 迁移率
    epsilon = 0.01     # 界面厚度参数
    lambda_param = epsilon**2


