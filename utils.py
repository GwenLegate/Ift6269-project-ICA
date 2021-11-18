import numpy as np

def tone_generator(duration, freq, fs):
    return np.sin(2*np.pi*np.arange(fs*duration)*freq/fs).astype(np.float64)