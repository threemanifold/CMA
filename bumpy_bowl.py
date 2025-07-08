import numpy as np


def bumpy_bowl(x):
    x = np.asarray(x)
    return (x[0]**2 + x[1]**2) / 20.0 + np.sin(x[0])**2 + np.sin(x[1])**2