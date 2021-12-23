import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def tone_generator(duration, freq, fs):
    return np.sin(2*np.pi*np.arange(fs*duration)*freq/fs).astype(np.float64)

def load_img(path, greyscale=True):
    if greyscale:
        img = Image.open(path).convert('L')
    else:
        img = Image.open(path)
    return np.array(img)

def show_img(data):
    data = data.clip(min=0)
    try:
        pilImage = Image.fromarray(data*255)
        pilImage.show()
    except:
        pilImage = Image.fromarray((data * 255).astype(np.uint8))
        pilImage.show()
