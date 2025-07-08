import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- definition -------------------------------------------------------------
def bumpy_bowl(x, y):
    """The Bumpy Bowl test surface f(x1,x2)."""
    return (x**2 + y**2) / 20.0 + np.sin(x)**2 + np.sin(y)**2

# ---- compute on a grid ------------------------------------------------------
x = np.linspace(-10, 10, 300)
y = np.linspace(-10, 10, 300)
X, Y = np.meshgrid(x, y)
Z = bumpy_bowl(X, Y)

# ---- 3-D surface plot -------------------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=5, cstride=5, linewidth=0)
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_zlabel(r"$f(\mathbf{x})$")
ax.set_title("Bumpy Bowl surface")

# ---- 2-D filled-contour plot -----------------------------------------------
fig2 = plt.figure()
cs = plt.contourf(X, Y, Z, levels=60)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.title("Bumpy Bowl contour")
plt.colorbar(cs)

plt.show()