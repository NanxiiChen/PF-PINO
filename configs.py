from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Configs:
    
    in_channels = 5 # phi, c, lp, mesh, time
    out_channels = 2 # phi, c
    modes = 8
    width = 64 # channel width in spectral conv layer
    depth = 4 # number of spectral conv layers
    activation = "gelu"
    
    learning_rate = 1e-3
    decay_every = 500
    decay_rate = 0.95
    end_value = 1e-5
    
    batch_size = 128
    epochs = 5000
    save_every = 100
    test_every = 500
    physical_residual = True
    
    save_dir = "./runs/PINO/" if physical_residual else "./runs/FNO/"
    data_dir = "./data/train_valid/"
    test_data_dir = "./data/test/"
    
    Lc = 1e-4 # xc = x / Lc
    Tc = 100.0 # tc = t / Tc
    # Lpc = lambda lp: -jnp.log10(lp) - 5  # lpc = -log10(lp) - 5
    
    @staticmethod
    def Lpc(lp):
        return -jnp.log10(lp) - 5  # lpc = -log10(lp) - 5
    
    @staticmethod
    def Lp_from_Lpc(lpc):
        return 10**(- (lpc + 5))
    AC_PRE_SCALE = 1e8
    CH_PRE_SCALE = 1e4
    
    
    ALPHA_PHI = 1.03e-4
    OMEGA_PHI = 1.76e7
    THICKNESS = 2.94 * jnp.sqrt(2 * ALPHA_PHI / OMEGA_PHI)
    MM = 7.94e-18
    DD = 8.5e-10
    AA = 5.35e7
    CSE = 1.0
    CLE = 5100 / 1.43e5
    
    
    
    
    