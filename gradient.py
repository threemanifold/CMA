import torch

def bumpy_bowl(x):
    # x is a tensor of shape (2,)
    return (x[0]**2 + x[1]**2) / 20.0 + torch.sin(x[0])**2 + torch.sin(x[1])**2

# --- choose initial point ----------------------------------------------------
# start at [5, 5], just like in the CMA-ES example
x = torch.tensor([5.0, 5.0], requires_grad=True)

# --- set up optimizer --------------------------------------------------------
optimizer = torch.optim.Adam([x], lr=0.1)

# --- optimization loop -------------------------------------------------------
n_steps = 2000
for step in range(1, n_steps+1):
    optimizer.zero_grad()
    loss = bumpy_bowl(x)
    loss.backward()
    optimizer.step()
    # keep x inside the original bounds [-10,10]
    with torch.no_grad():
        x.clamp_(-10.0, 10.0)

    # log every 200 steps
    if step % 200 == 0 or step == 1:
        print(f"step {step:4d} → x = {x.detach().numpy()},  f(x) = {loss.item():.6f}")

# --- final result ------------------------------------------------------------
print("\nFinal solution:")
print(f"  x = {x.detach().numpy()}")
print(f"  f(x) = {bumpy_bowl(x).item():.6f}")