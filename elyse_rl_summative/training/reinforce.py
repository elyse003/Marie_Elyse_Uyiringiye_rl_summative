"""
training/reinforce.py
=====================
Vanilla REINFORCE (Monte-Carlo Policy Gradient) — implemented
from scratch in PyTorch since Stable Baselines3 does not ship
a REINFORCE class.

Algorithm
---------
  1. Roll out full episode using current policy π_θ
  2. Compute discounted returns G_t for each timestep
  3. (Optionally) normalise returns
  4. Update θ ← θ + α ∑_t G_t ∇_θ log π_θ(a_t | s_t)
  5. Repeat

Hyperparameter space (10 configs):
  learning_rate, gamma, hidden_sizes,
  normalize_returns, entropy_coef, grad_clip
"""

from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import CropDiseaseEnv

MODEL_DIR  = os.path.join("models", "pg")
RESULT_DIR = "results"
PLOT_DIR   = os.path.join("results", "plots")
CSV_PATH   = os.path.join(RESULT_DIR, "reinforce_results.csv")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Policy network ────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)

    def select_action(self, obs: np.ndarray):
        x    = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
        probs = self(x).squeeze(0)
        dist  = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()


# ── 10 Hyperparameter configs ─────────────────────────────────────────────────

REINFORCE_CONFIGS = [
    # Run 0 — baseline
    dict(learning_rate=1e-3, gamma=0.99, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=1.0),

    # Run 1 — lower LR
    dict(learning_rate=5e-4, gamma=0.99, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=1.0),

    # Run 2 — no return normalisation
    dict(learning_rate=1e-3, gamma=0.99, hidden_sizes=[128, 128],
         normalize_returns=False, entropy_coef=0.01, grad_clip=1.0),

    # Run 3 — high entropy bonus (more exploration)
    dict(learning_rate=1e-3, gamma=0.99, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.05, grad_clip=1.0),

    # Run 4 — no entropy bonus
    dict(learning_rate=1e-3, gamma=0.95, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.00, grad_clip=1.0),

    # Run 5 — deeper network
    dict(learning_rate=5e-4, gamma=0.99, hidden_sizes=[256, 256, 128],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=0.5),

    # Run 6 — shallow wide
    dict(learning_rate=1e-3, gamma=0.99, hidden_sizes=[512],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=1.0),

    # Run 7 — low gamma (short-sighted)
    dict(learning_rate=1e-3, gamma=0.90, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=1.0),

    # Run 8 — very small LR
    dict(learning_rate=1e-4, gamma=0.99, hidden_sizes=[128, 128],
         normalize_returns=True,  entropy_coef=0.01, grad_clip=1.0),

    # Run 9 — no grad clipping
    dict(learning_rate=1e-3, gamma=0.99, hidden_sizes=[128, 64],
         normalize_returns=True,  entropy_coef=0.02, grad_clip=None),
]


# ── REINFORCE trainer ─────────────────────────────────────────────────────────

def train_reinforce(
    config_idx: int,
    n_episodes: int = 1000,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    cfg = REINFORCE_CONFIGS[config_idx]
    run_tag = f"reinforce_run_{config_idx}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  REINFORCE Run {config_idx:>2} / {len(REINFORCE_CONFIGS)-1}")
        print(f"  lr={cfg['learning_rate']}, gamma={cfg['gamma']}, "
              f"hidden={cfg['hidden_sizes']}")
        print(f"  norm_returns={cfg['normalize_returns']}, "
              f"entropy={cfg['entropy_coef']}, "
              f"grad_clip={cfg['grad_clip']}")
        print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CropDiseaseEnv(grid_size=8, max_steps=250)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    policy    = PolicyNetwork(obs_dim, act_dim, cfg["hidden_sizes"]).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

    episode_rewards = []
    episode_losses  = []
    entropy_log     = []

    t0 = time.time()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        log_probs, entropies, rewards = [], [], []

        while True:
            action, log_prob, entropy = policy.select_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            if terminated or truncated:
                break

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)

        # ── Compute discounted returns ─────────────────────────────────
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + cfg["gamma"] * G
            returns.insert(0, G)

        returns_t = torch.FloatTensor(returns).to(DEVICE)
        if cfg["normalize_returns"] and returns_t.std() > 1e-8:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── Policy loss ────────────────────────────────────────────────
        log_probs_t  = torch.stack(log_probs)
        entropies_t  = torch.stack(entropies)
        policy_loss  = -(log_probs_t * returns_t).sum()
        entropy_loss = -cfg["entropy_coef"] * entropies_t.mean()
        loss         = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        if cfg["grad_clip"] is not None:
            nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"])
        optimizer.step()

        episode_losses.append(loss.item())
        entropy_log.append(entropies_t.mean().item())

        if verbose and (ep + 1) % 100 == 0:
            recent_mean = np.mean(episode_rewards[-20:])
            print(f"  Ep {ep+1:>5}/{n_episodes}  "
                  f"mean_reward={recent_mean:>+8.2f}  "
                  f"loss={loss.item():>8.4f}  "
                  f"entropy={entropy_log[-1]:.4f}")

    elapsed = time.time() - t0
    env.close()

    # Save model weights
    model_path = os.path.join(MODEL_DIR, f"{run_tag}.pt")
    torch.save(policy.state_dict(), model_path)

    mean_r = float(np.mean(episode_rewards[-50:])) if len(episode_rewards) >= 50 else float(np.mean(episode_rewards))
    std_r  = float(np.std(episode_rewards[-50:]))  if len(episode_rewards) >= 50 else 0.0
    max_r  = float(np.max(episode_rewards))

    if verbose:
        print(f"  ✓  Run {config_idx} done in {elapsed:.1f}s | "
              f"mean_ep_reward={mean_r:.1f} ± {std_r:.1f}")

    return {
        "run":               config_idx,
        "learning_rate":     cfg["learning_rate"],
        "gamma":             cfg["gamma"],
        "hidden_sizes":      str(cfg["hidden_sizes"]),
        "normalize_returns": cfg["normalize_returns"],
        "entropy_coef":      cfg["entropy_coef"],
        "grad_clip":         cfg["grad_clip"],
        "mean_ep_reward":    round(mean_r, 2),
        "std_ep_reward":     round(std_r,  2),
        "max_ep_reward":     round(max_r,  2),
        "n_episodes":        len(episode_rewards),
        "train_time_s":      round(elapsed, 1),
        "episode_rewards":   episode_rewards,
        "losses":            episode_losses,
        "entropies":         entropy_log,
    }


# ── Save / plot ───────────────────────────────────────────────────────────────

def save_csv(results: list[dict]):
    keys = [k for k in results[0].keys()
            if k not in ("episode_rewards", "losses", "entropies")]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"  Results CSV → {CSV_PATH}")


def plot_results(results: list[dict]):
    n = len(results)
    cols = min(5, n);  rows = (n + cols - 1) // cols

    # Reward subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle("REINFORCE — Episode Reward per Run", fontsize=14, fontweight="bold")
    for i, r in enumerate(results):
        ax = axes[i]
        rew = r["episode_rewards"]
        ax.plot(rew, alpha=0.3, color="coral", linewidth=0.7)
        w = min(30, len(rew))
        if len(rew) >= w:
            sm = np.convolve(rew, np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(rew)), sm, color="darkred", linewidth=1.8)
        ax.set_title(
            f"Run {i}  lr={r['learning_rate']}  γ={r['gamma']}\n"
            f"norm={r['normalize_returns']}  ent={r['entropy_coef']}",
            fontsize=8,
        )
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward",  fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[n:]:
        ax.set_visible(False)
    p = os.path.join(PLOT_DIR, "reinforce_reward_curves.png")
    fig.savefig(p, dpi=130, bbox_inches="tight");  plt.close(fig)
    print(f"  Reward curves → {p}")

    # Entropy curves
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for r in results:
        ent = r["entropies"]
        w   = min(30, len(ent))
        if len(ent) >= w:
            sm = np.convolve(ent, np.ones(w)/w, mode="valid")
            ax2.plot(sm, alpha=0.7, linewidth=0.9, label=f"Run {r['run']}")
    ax2.set_title("REINFORCE — Policy Entropy Curves", fontweight="bold")
    ax2.set_xlabel("Episode");  ax2.set_ylabel("Entropy")
    ax2.legend(fontsize=7, ncol=5)
    fig2.tight_layout()
    p2 = os.path.join(PLOT_DIR, "reinforce_entropy.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight");  plt.close(fig2)
    print(f"  Entropy curves → {p2}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--run_id",   type=int, default=-1)
    parser.add_argument("--all",      action="store_true")
    args = parser.parse_args()

    run_ids = list(range(len(REINFORCE_CONFIGS))) \
              if (args.all or args.run_id < 0) else [args.run_id]

    results = []
    for idx in run_ids:
        r = train_reinforce(idx, n_episodes=args.episodes)
        results.append(r)

    save_csv(results)
    plot_results(results)

    best = max(results, key=lambda x: x["mean_ep_reward"])
    print(f"\n  Best run: {best['run']} — mean_ep_reward={best['mean_ep_reward']}")


if __name__ == "__main__":
    main()
