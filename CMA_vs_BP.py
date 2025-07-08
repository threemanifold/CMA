import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import cma

# -- target function -----------------------------------------------------------
def target_fn(x):
    return np.sin(2*np.pi*x) + 0.3*np.cos(3*np.pi*x)

# -- dataset ------------------------------------------------------------------
np.random.seed(0)
x_train = np.random.uniform(-1, 1, 200)
y_train = target_fn(x_train)
x_train_t = torch.from_numpy(x_train).float().unsqueeze(1)
y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)

# -- neural network -----------------------------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(10, 1)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

# -- backprop training -------------------------------------------------------
model_bp = Net()
optimizer = optim.Adam(model_bp.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

print("Training with backprop...")
for epoch in range(1, 1001):
    optimizer.zero_grad()
    y_pred = model_bp(x_train_t)
    loss = loss_fn(y_pred, y_train_t)
    loss.backward()
    optimizer.step()
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d}, MSE = {loss.item():.6f}")
print("Backprop final MSE:", loss.item())

# -- prepare model for CMA-ES ------------------------------------------------
model_cma = Net()
#model_cma.load_state_dict(model_bp.state_dict())  # start from backprop solution
param_shapes = [p.shape for p in model_cma.parameters()]

def get_params_vector(net):
    return np.concatenate([p.detach().numpy().ravel() for p in net.parameters()])

def set_params_vector(net, vec):
    idx = 0
    for p, shape in zip(net.parameters(), param_shapes):
        size = int(np.prod(shape))
        vals = vec[idx:idx+size].reshape(shape)
        p.data = torch.from_numpy(vals.astype(np.float32))
        idx += size

# -- CMA-ES optimization -----------------------------------------------------
x0 = get_params_vector(model_cma)
es = cma.CMAEvolutionStrategy(x0, 0.5, {'maxiter': 1000})

def cma_objective(v):
    set_params_vector(model_cma, v)
    with torch.no_grad():
        y_pred = model_cma(x_train_t).numpy().flatten()
    return float(np.mean((y_pred - y_train)**2))

print("Optimizing with CMA-ES...")
es.optimize(cma_objective)

# -- results ------------------------------------------------------------------
# backprop predictions
y_line = np.linspace(-1, 1, 300)
y_true = target_fn(y_line)
y_input = torch.from_numpy(y_line).float().unsqueeze(1)
with torch.no_grad():
    y_bp = model_bp(y_input).numpy().flatten()
    y_cma = model_cma(y_input).numpy().flatten()

# plot comparison
plt.figure()
plt.plot(y_line, y_true, 'k-', label='True')
plt.plot(y_line, y_bp, 'b--', label='Backprop')
plt.plot(y_line, y_cma, 'r-.', label='CMA-ES')
plt.legend()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function approximation: Backprop vs CMA-ES')
plt.show()

# final MSE printout
mse_bp = float(loss_fn(model_bp(x_train_t), y_train_t).item())
mse_cma = float(np.mean((y_cma - y_train)**2))
print(f"Final MSE - Backprop: {mse_bp:.6f}, CMA-ES: {mse_cma:.6f}")
