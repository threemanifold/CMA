import argparse
from typing import List

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cma

# ---------------------------------------------------------------------------
# Policy network (tiny MLP) --------------------------------------------------
# ---------------------------------------------------------------------------
class PolicyNet(nn.Module):
    """A minimal 2‑layer neural net policy for CartPole‑v1."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)  # two discrete actions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # logits

    # ---------------------------------------------------------------------
    # Convenience: choose an action ---------------------------------------
    # ---------------------------------------------------------------------
    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        """Return an action for a single CartPole state array (shape (4,))."""
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 4)
        logits = self.forward(state_t)
        if greedy:
            action = torch.argmax(logits, dim=1)
        else:
            probs = torch.softmax(logits, dim=1)
            action = torch.distributions.Categorical(probs).sample()
        return int(action.item())

# ---------------------------------------------------------------------------
# Episode rollout -----------------------------------------------------------
# ---------------------------------------------------------------------------

def rollout(
    env: gym.Env,
    policy: PolicyNet,
    *,
    greedy: bool = False,
    render: bool = False,
    max_steps: int | None = None,
) -> float:
    """Run **one** episode and return cumulative reward."""
    state, _ = env.reset()
    episode_reward = 0.0
    steps = 0
    done = False
    while not done:
        if render:
            env.render()
        action = policy.act(state, greedy=greedy)
        state, reward, done, truncated, _ = env.step(action)
        episode_reward += reward
        steps += 1
        if truncated or (max_steps and steps >= max_steps):
            break
    return episode_reward

# ---------------------------------------------------------------------------
# Helpers to flatten / restore network weights ------------------------------
# ---------------------------------------------------------------------------

def params_to_vector(net: nn.Module) -> np.ndarray:
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in net.parameters()])


def vector_to_params(net: nn.Module, vec: np.ndarray) -> None:
    idx = 0
    for p in net.parameters():
        size = p.numel()
        block = vec[idx : idx + size].reshape(p.shape)
        p.data = torch.from_numpy(block).float()
        idx += size

# ---------------------------------------------------------------------------
# Main optimisation routine --------------------------------------------------
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="CMA‑ES policy search for CartPole‑v1")
    parser.add_argument("--popsize", type=int, default=None, help="Population size λ (optional)")
    parser.add_argument("--sigma0", type=float, default=0.5, help="Initial CMA step‑size σ₀")
    parser.add_argument("--rollouts", type=int, default=4, help="Number of episodes to average per evaluation")
    parser.add_argument("--maxiter", type=int, default=200, help="Max CMA generations (iterations)")
    parser.add_argument("--render", action="store_true", help="Render the final greedy episode")
    args = parser.parse_args()

    # Gym environment (note: render only in the final demo to save CPU)
    render_mode = "human" if args.render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Fresh policy + initial parameter vector
    policy = PolicyNet()
    x0 = params_to_vector(policy)

    # Objective = negative average reward over N rollouts (because CMA minimises)
    def objective(vec: List[float]) -> float:  # CMA passes a Python list
        vec_np = np.asarray(vec, dtype=np.float32)
        vector_to_params(policy, vec_np)
        rewards = [rollout(env, policy, greedy=False, render=False) for _ in range(args.rollouts)]
        return -float(np.mean(rewards))

    # CMA‑ES set‑up ----------------------------------------------------------
    inopts = {}
    if args.popsize is not None:
        inopts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, inopts)

    print("Starting CMA‑ES optimisation …")
    es.optimize(objective, args.maxiter)

    # Results ---------------------------------------------------------------
    best_reward = -es.result.fbest  # negate because we minimised negative reward
    print(f"Best average reward (over {args.rollouts} rollouts): {best_reward:.1f}")

    # Load best params back into policy and run a greedy episode -------------
    vector_to_params(policy, es.result.xbest)
    greedy_reward = rollout(env, policy, greedy=True, render=args.render)
    print(f"Greedy reward from best policy: {greedy_reward}")
    env.close()


if __name__ == "__main__":
    main()
