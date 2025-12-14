from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    DEBUG_MODE = False

    # Model architecture settings
    model_type = "fno"  # Options: 'fno', 'fcn', 'unet'

    in_channels = 5 # phi, c, meshx, meshy, dt
    out_channels = 2 # phi, c
    modes_x = 8
    modes_y = 8
    width = 64
    depth = 4
    activation = "relu"
    down_scale = 2


    learning_rate = 5e-4
    decay_every = 200
    decay_rate = 0.95
    end_value = 1e-5
    

    batch_size = 128
    epochs = 5000
    save_every = 100
    test_every = 500
    physical_residual = True

    save_dir = f"/root/autodl-tmp/runs/corrosion2d/FNO-DS2-PI/"
    if DEBUG_MODE:
        save_dir = save_dir[:-1] + "_DEBUG/"
    data_dir = "/root/autodl-tmp/data/corrosion2d/train_valid/"
    test_data_dir = "/root/autodl-tmp/data/corrosion2d/test/"
    
    Lc = 1e-4 # xc = x / Lc
    Tc = 100.0 # tc = t / Tc


    AC_PRE_SCALE = 1e0
    CH_PRE_SCALE = 1e4

    ALPHA_PHI = 9.62e-5/4
    OMEGA_PHI = 1.663e7
    THICKNESS = 2.94 * jnp.sqrt(2 * ALPHA_PHI / OMEGA_PHI)
    MM = 7.94e-18
    DD = 8.5e-10
    AA = 5.35e7
    Lp = 1.0e-10
    CSE = 1.0
    CLE = 5100 / 1.43e5


