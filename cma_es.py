import cma
import numpy as np


def bumpy_bowl(x):
    x = np.asarray(x)
    return (x[0]**2 + x[1]**2) / 20.0 + np.sin(x[0])**2 + np.sin(x[1])**2

# --- CMA-ES run --------------------------------------------------------------
es = cma.CMAEvolutionStrategy(x0=[ 5, 5],    # any starting point in [-10,10]^2
                              sigma0=2.0,    # initial step-size
                              inopts={'bounds': [-10, 10]})

es.optimize(bumpy_bowl)            # ← main loop
print("Best found x, f(x):", es.result.xbest, es.result.fbest)