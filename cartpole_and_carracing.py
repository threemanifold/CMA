"""cma_gym_parallel.py               # <<< renamed (old name still works as a symlink if you like)

Parallel CMA-ES for CartPole-v1 **or** CarRacing-v3.
 - Choose the environment with --env {cartpole|carracing}
 - MLP policy is used for CartPole, small CNN for CarRacing.
 - CarRacing frames are (optionally) gray-scaled, resized and frame-stacked
   via standard Gymnasium wrappers (GrayScaleObservation, ResizeObservation,
   FrameStack). :contentReference[oaicite:0]{index=0}
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from typing import List, Callable, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cma


# ---------------------------------------------------------------------------
# Policy networks ------------------------------------------------------------
# ---------------------------------------------------------------------------
class PolicyMLP(nn.Module):
    """Minimal 2-layer MLP policy for CartPole."""
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # logits
        return self.fc2(F.relu(self.fc1(x)))

    def act(self, state: np.ndarray, greedy: bool = False) -> int:
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        logits = self.forward(state_t)
        if greedy:
            return int(torch.argmax(logits, dim=1).item())
        probs = torch.softmax(logits, dim=1)
        return int(torch.distributions.Categorical(probs).sample().item())


class PolicyCNN(nn.Module):
    """Tiny convolutional policy for CarRacing with *continuous* actions."""
    def __init__(self, in_frames: int = 4):                       # <<< n_actions removed
        super().__init__()
        self.conv1 = nn.Conv2d(in_frames, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(64 * 2 * 2, 128)
        self.fc2   = nn.Linear(128, 3)                            # <<< steer, gas, brake

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x / 255.0))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(self.flat(x)))
        return self.fc2(x)                                        # raw continuous heads

    def act(self, state: np.ndarray, greedy: bool = False) -> np.ndarray:
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        out = self.forward(st)
        # map heads to action bounds -----------------------------
        steer  = torch.tanh(out[:, 0])            # [-1 ,  1]
        gas    = torch.sigmoid(out[:, 1])         # [ 0 ,  1]
        brake  = torch.sigmoid(out[:, 2])         # [ 0 ,  1]
        action = torch.cat([steer, gas, brake], dim=1)
        return action.squeeze(0).cpu().numpy()    # shape (3,)


# ---------------------------------------------------------------------------
# Environment builder --------------------------------------------------------
# ---------------------------------------------------------------------------
def make_env(env_name: str, render_mode=None, frame_stack: int = 4):
    if env_name == "cartpole":
        return gym.make("CartPole-v1", render_mode=render_mode)

    if env_name == "carracing":
        env = gym.make("CarRacing-v3", render_mode=render_mode)   # <<< default = continuous
        from gymnasium.wrappers import GrayscaleObservation, ResizeObservation, FrameStackObservation
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, (48, 48))
        env = FrameStackObservation(env, frame_stack)
        return env
    raise ValueError(f"Unknown env '{env_name}'")


# ---------------------------------------------------------------------------
# Rollout --------------------------------------------------------------------
# ---------------------------------------------------------------------------
def rollout(policy_params, rollouts_per_eval, env_name, env_seed=None) -> float:
    env = make_env(env_name)
    if env_seed is not None:
        env.reset(seed=env_seed)

    if env_name == "cartpole":
        policy = PolicyMLP()
    else:
        policy = PolicyCNN(env.observation_space.shape[0])        # <<< no n_actions

    _vector_to_params(policy, policy_params)

    returns = []
    for _ in range(rollouts_per_eval):
        s, _ = env.reset()
        term = trunc = False
        ret = 0.0
        while not (term or trunc):
            a = policy.act(s)
            s, r, term, trunc, _ = env.step(a)
            ret += r
        returns.append(ret)
    env.close()
    return -float(np.mean(returns))


# ---------------------------------------------------------------------------
# Parameter helpers ----------------------------------------------------------
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
# Main loop ------------------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel CMA-ES on CartPole or CarRacing")
    parser.add_argument("--env", choices=["cartpole", "carracing"], default="cartpole")   # <<<
    parser.add_argument("--popsize", type=int, default=None, help="Population size λ")
    parser.add_argument("--sigma0",  type=float, default=0.5, help="Initial CMA step-size σ₀")
    parser.add_argument("--rollouts", type=int, default=4, help="Episodes averaged per offspring")
    parser.add_argument("--maxiter", type=int, default=200, help="Max generations")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Parallel worker processes")
    parser.add_argument("--render",  action="store_true", help="Render a final greedy episode")
    args = parser.parse_args()

    # pick seed policy arch
    if args.env == "cartpole":
        seed_policy = PolicyMLP()
    else:
        dummy_env = make_env("carracing")
        seed_policy = PolicyCNN(dummy_env.observation_space.shape[0])  # <<< no n_actions
        dummy_env.close()

    # ---------------------------------------------------------------------
    # Prepare CMA-ES
    # ---------------------------------------------------------------------
    x0 = _params_to_vector(seed_policy)
    es_opts: dict[str, int] = {}
    if args.popsize is not None:
        es_opts["popsize"] = args.popsize
    es = cma.CMAEvolutionStrategy(x0, args.sigma0, es_opts)

    # Multiprocessing pool -------------------------------------------------
    mp_context = mp.get_context("spawn")                      # macOS / PyTorch safe
    with mp_context.Pool(processes=args.workers) as pool:
        print(f"Using {args.workers} worker processes …")
        for gen in range(args.maxiter):
            offspring = es.ask()                             # 1) sample
            fitnesses = pool.starmap(                        # 2) parallel eval
                rollout,
                [(np.asarray(vec, np.float32),
                  args.rollouts,
                  args.env,
                  None) for vec in offspring],               # env passed in
            )
            es.tell(offspring, fitnesses)                    # 3) update
            if gen % 10 == 0:
                es.disp()
            if es.stop():
                print("Stopping criteria met.")
                break

    # ---------------------------------------------------------------------
    # Best solution & greedy check
    # ---------------------------------------------------------------------
    best_vec = es.best.x
    best_avg_reward = -es.best.f
    print(f"Best average reward over {args.rollouts} rollouts: {best_avg_reward:.1f}")

    # greedy run
    env_test = make_env(args.env, render_mode="human" if args.render else None)
    if args.env == "cartpole":
        policy_best = PolicyMLP()
    else:
        policy_best = PolicyCNN(env_test.observation_space.shape[0])  # <<< no n_actions
    _vector_to_params(policy_best, best_vec)

    state, _ = env_test.reset()
    term = trunc = False
    total_r = 0.0
    while not (term or trunc):
        action = policy_best.act(state, greedy=True)
        state, r, term, trunc, _ = env_test.step(action)
        total_r += r
    env_test.close()
    print(f"Greedy reward in single episode: {total_r}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()