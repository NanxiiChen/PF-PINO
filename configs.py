from dataclasses import dataclass

@dataclass(frozen=True)
class Configs:
    
    in_channels = 5 # phi, c, lp, mesh, time
    out_channels = 2 # phi, c
    modes = 32
    width = 128 # channel width in spectral conv layer
    depth = 4 # number of spectral conv layers
    activation = "gelu"
    
    learning_rate = 1e-3
    decay_every = 100
    decay_rate = 0.95
    end_value = 1e-5
    
    batch_size = 64
    epochs = 2000
    save_every = 100
    
    save_dir = "./runs/FNO/"
    data_dir = "./data/train_valid/"
    test_data_dir = "./data/test/"
    
    
    
    
    