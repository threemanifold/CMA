"""cma_cartpole_parallel.py

Parallel CMA‑ES for CartPole‑v1.
Evaluates offspring in parallel across CPU cores using multiprocessing.Pool.

Usage (examples):
    # default popsize, auto‑detected workers
    python cma_cartpole_parallel.py

    # custom population, 8 workers, render final run
    python cma_cartpole_parallel.py --popsize 32 --workers 8 --render
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import List

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cma

# ---------------------------------------------------------------------------
# Policy network (tiny MLP) --------------------------------------------------
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """Minimal 2‑layer MLP policy for CartPole."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_size) # 4 --> 32 + 256 + cell_state=256
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # logits
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    # ---------------------------------------------------------------------
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Select action 0/1 given single CartPole state (shape (4,))."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(state_t)
        if greedy:
            action = torch.argmax(logits, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            action = torch.distributions.Categorical(probs).sample()
        return int(action.item())

# ---------------------------------------------------------------------------
# Rollout --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def rollout(
    policy_params: np.ndarray,
    rollouts_per_eval: int,
    env_seed: int | None = None,
) -> float: # TODO: this can be done in parallel as well
    """Return **negative** average reward (for minimisation) for one offspring.

    A fresh Gym env and PolicyNet are built inside the worker; this avoids
    pickling large objects between processes and keeps each worker independent.
    """
    # Build environment – no rendering in workers
    env = gym.make("CartPole-v1", render_mode=None)
    #env = gym.make("CarRacing-v3", render_mode=None)
    if env_seed is not None:
        env.reset(seed=env_seed)

    # Build policy and load parameters
    policy = PolicyNet()
    _vector_to_params(policy, policy_params)

    rewards = []
    for _ in range(rollouts_per_eval):
        state, _ = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = policy.act(state, greedy=False)
            state, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            if truncated:
                break
        rewards.append(ep_reward)
    env.close()

    return -float(np.mean(rewards))  # CMA‑ES minimises

# ---------------------------------------------------------------------------
# Weight‑vector helpers ------------------------------------------------------
# ---------------------------------------------------------------------------

def _params_to_vector(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def _vector_to_params(net: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    for p in net.parameters():
        numel = p.numel()
        block = vec[idx : idx + numel].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        idx += numel

# ---------------------------------------------------------------------------
# Main optimisation loop -----------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel CMA‑ES on CartPole‑v1")
    parser.add_argument("--popsize", type=int, default=None, help="Population size λ")
    parser.add_argument("--sigma0", type=float, default=0.5, help="Initial CMA step‑size σ₀")
    parser.add_argument("--rollouts", type=int, default=4, help="Episodes averaged per offspring")
    parser.add_argument("--maxiter", type=int, default=200, help="Max generations")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker processes")
    parser.add_argument("--render", action="store_true", help="Render a final greedy episode")
    args = parser.parse_args()

    # ---------------------------------------------------------------------
    # Prepare initial individual and CMA object ---------------------------
    # ---------------------------------------------------------------------
    seed_policy = PolicyNet()
    x0 = _params_to_vector(seed_policy)
    inopts: dict[str, int] = {}
    if args.popsize is not None:
        inopts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, inopts)

    # Multiprocessing pool -------------------------------------------------
    mp_context = mp.get_context("spawn")  # safer on macOS / PyTorch
    with mp_context.Pool(processes=args.workers) as pool:
        print(f"Using {args.workers} worker processes …")
        for generation in range(args.maxiter):
            # 1) sample offspring
            offspring = es.ask()
            # 2) evaluate in parallel – pool.map returns list in same order
            fitnesses = pool.starmap(
                rollout,
                [(np.asarray(vec, dtype=np.float32), args.rollouts, None) for vec in offspring],
            )
            # 3) update CMA‑ES with fitnesses
            es.tell(offspring, fitnesses)
            if generation % 10 == 0:
                es.disp()
            if es.stop():
                print("Stopping criteria met.")
                break

        # ---------------------------------------------------------------------
    # Retrieve best solution -------------------------------------------------
    # ---------------------------------------------------------------------
    # es.best is a small helper object; its fields are .x (solution vector), .f (objective)
    best_vec = es.best.x  # numpy array of parameters
    best_f = es.best.f    # minimised objective = negative average reward

    best_avg_reward = -best_f
    print(f"Best average reward over {args.rollouts} rollouts: {best_avg_reward:.1f}")

    # Test greedy performance (optional render) -----------------------------
    policy_best = PolicyNet()
    _vector_to_params(policy_best, best_vec)
    env_test = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    state, _ = env_test.reset()
    done = False
    tot_reward = 0.0
    while not done:
        action = policy_best.act(state, greedy=True)
        state, reward, done, truncated, _ = env_test.step(action)
        tot_reward += reward
        if truncated:
            break
    env_test.close()
    print(f"Greedy reward in single episode: {tot_reward}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()