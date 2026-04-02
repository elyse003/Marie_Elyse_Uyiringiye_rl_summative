"""
training/dqn_training.py
========================
Trains a Deep Q-Network (DQN) agent on CropDiseaseEnv using
Stable Baselines 3.  Runs 10 hyperparameter configurations and
logs results to CSV + plots.

Run:
    python -m training.dqn_training [--timesteps 300000] [--run_id 0]
    python -m training.dqn_training --all          # trains all 10 configs

Outputs
-------
  models/dqn/dqn_run_<N>.zip          — saved model
  results/dqn_results.csv             — hyperparameter + metric table
  results/plots/dqn_reward_curves.png — learning curves
  results/plots/dqn_loss_curves.png   — TD-loss curves
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import CropDiseaseEnv

# ── Output paths ──────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join("models", "dqn")
RESULT_DIR = "results"
PLOT_DIR   = os.path.join("results", "plots")
CSV_PATH   = os.path.join(RESULT_DIR, "dqn_results.csv")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ── 10 Hyperparameter configurations ─────────────────────────────────────────
#
#   Columns: learning_rate, gamma, batch_size, buffer_size,
#            exploration_fraction, exploration_final_eps,
#            target_update_interval, train_freq, gradient_steps,
#            policy_kwargs (net_arch)
#
DQN_CONFIGS = [
    # Run 0 — baseline
    dict(learning_rate=1e-3, gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_fraction=0.15, exploration_final_eps=0.05,
         target_update_interval=500,  train_freq=4,  gradient_steps=1,
         net_arch=[128, 128]),

    # Run 1 — lower LR, bigger buffer
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64,  buffer_size=100_000,
         exploration_fraction=0.20, exploration_final_eps=0.05,
         target_update_interval=500,  train_freq=4,  gradient_steps=1,
         net_arch=[128, 128]),

    # Run 2 — high LR, small buffer
    dict(learning_rate=2e-3, gamma=0.95, batch_size=32,  buffer_size=10_000,
         exploration_fraction=0.10, exploration_final_eps=0.10,
         target_update_interval=300,  train_freq=4,  gradient_steps=1,
         net_arch=[64, 64]),

    # Run 3 — deeper network
    dict(learning_rate=1e-3, gamma=0.99, batch_size=128, buffer_size=50_000,
         exploration_fraction=0.20, exploration_final_eps=0.05,
         target_update_interval=1000, train_freq=4,  gradient_steps=1,
         net_arch=[256, 256]),

    # Run 4 — longer exploration
    dict(learning_rate=5e-4, gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_fraction=0.30, exploration_final_eps=0.02,
         target_update_interval=500,  train_freq=4,  gradient_steps=1,
         net_arch=[128, 128]),

    # Run 5 — low gamma (myopic)
    dict(learning_rate=1e-3, gamma=0.90, batch_size=64,  buffer_size=50_000,
         exploration_fraction=0.15, exploration_final_eps=0.05,
         target_update_interval=500,  train_freq=4,  gradient_steps=1,
         net_arch=[128, 64]),

    # Run 6 — large batch, more gradient steps
    dict(learning_rate=3e-4, gamma=0.99, batch_size=256, buffer_size=100_000,
         exploration_fraction=0.20, exploration_final_eps=0.02,
         target_update_interval=1000, train_freq=8,  gradient_steps=4,
         net_arch=[256, 128]),

    # Run 7 — very small LR, slow convergence test
    dict(learning_rate=1e-4, gamma=0.99, batch_size=64,  buffer_size=50_000,
         exploration_fraction=0.25, exploration_final_eps=0.05,
         target_update_interval=500,  train_freq=4,  gradient_steps=1,
         net_arch=[128, 128]),

    # Run 8 — aggressive target update
    dict(learning_rate=1e-3, gamma=0.98, batch_size=64,  buffer_size=50_000,
         exploration_fraction=0.15, exploration_final_eps=0.05,
         target_update_interval=200,  train_freq=4,  gradient_steps=2,
         net_arch=[128, 128]),

    # Run 9 — wide shallow network
    dict(learning_rate=5e-4, gamma=0.99, batch_size=128, buffer_size=75_000,
         exploration_fraction=0.20, exploration_final_eps=0.03,
         target_update_interval=750,  train_freq=4,  gradient_steps=1,
         net_arch=[512]),
]


# ── Callback: collect episode rewards + losses ────────────────────────────────

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.losses: list[float] = []
        self._ep_reward = 0.0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        # Read loss from SB3's internal logger
        try:
            loss = self.model.logger.name_to_value.get("train/loss", None)
            if loss is not None:
                self.losses.append(float(loss))
        except Exception:
            pass
        return True


# ── Train one configuration ───────────────────────────────────────────────────

def train_dqn(
    config_idx: int,
    timesteps: int = 100_000,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    cfg = DQN_CONFIGS[config_idx]
    run_tag = f"dqn_run_{config_idx}"
    print(f"\n{'='*60}")
    print(f"  DQN Run {config_idx:>2} / {len(DQN_CONFIGS)-1}")
    print(f"  Config: lr={cfg['learning_rate']}, gamma={cfg['gamma']}, "
          f"batch={cfg['batch_size']}, buf={cfg['buffer_size']}")
    print(f"  Net:    {cfg['net_arch']}    expl={cfg['exploration_fraction']}")
    print(f"{'='*60}")

    def make_env():
        e = CropDiseaseEnv(grid_size=8, max_steps=250)
        return Monitor(e)

    env      = make_vec_env(make_env, n_envs=1, seed=seed)
    eval_env = make_vec_env(make_env, n_envs=1, seed=seed + 100)

    callback    = MetricsCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=RESULT_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        exploration_fraction=cfg["exploration_fraction"],
        exploration_final_eps=cfg["exploration_final_eps"],
        target_update_interval=cfg["target_update_interval"],
        train_freq=cfg["train_freq"],
        gradient_steps=cfg["gradient_steps"],
        policy_kwargs={"net_arch": cfg["net_arch"]},
        seed=seed,
        verbose=verbose,
        tensorboard_log=None,
    )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[callback, eval_callback],
        progress_bar=False,
    )
    elapsed = time.time() - t0

    model_path = os.path.join(MODEL_DIR, f"{run_tag}.zip")
    model.save(model_path)

    ep_rewards = callback.episode_rewards
    mean_r     = float(np.mean(ep_rewards[-20:])) if len(ep_rewards) >= 20 else float(np.mean(ep_rewards)) if ep_rewards else 0.0
    std_r      = float(np.std(ep_rewards[-20:]))  if len(ep_rewards) >= 20 else 0.0
    max_r      = float(np.max(ep_rewards))         if ep_rewards else 0.0

    print(f"  ✓  Run {config_idx} done in {elapsed:.1f}s | "
          f"mean_ep_reward={mean_r:.1f} ± {std_r:.1f}")

    env.close();  eval_env.close()

    return {
        "run":                   config_idx,
        "learning_rate":         cfg["learning_rate"],
        "gamma":                 cfg["gamma"],
        "batch_size":            cfg["batch_size"],
        "buffer_size":           cfg["buffer_size"],
        "exploration_fraction":  cfg["exploration_fraction"],
        "exploration_final_eps": cfg["exploration_final_eps"],
        "target_update_interval":cfg["target_update_interval"],
        "net_arch":              str(cfg["net_arch"]),
        "mean_ep_reward":        round(mean_r, 2),
        "std_ep_reward":         round(std_r,  2),
        "max_ep_reward":         round(max_r,  2),
        "n_episodes":            len(ep_rewards),
        "train_time_s":          round(elapsed, 1),
        "episode_rewards":       ep_rewards,   # not written to CSV
        "losses":                callback.losses,
    }


# ── Save CSV results ──────────────────────────────────────────────────────────

def save_csv(results: list[dict]):
    keys = [k for k in results[0].keys()
            if k not in ("episode_rewards", "losses")]
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"\n  Results CSV → {CSV_PATH}")


# ── Plot learning curves ──────────────────────────────────────────────────────

def plot_results(results: list[dict]):
    n = len(results)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    # Reward curves
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle("DQN — Episode Reward per Run", fontsize=14, fontweight="bold")

    for i, r in enumerate(results):
        ax = axes[i]
        rewards = r["episode_rewards"]
        if rewards:
            ax.plot(rewards, alpha=0.4, color="steelblue", linewidth=0.8)
            window = min(20, len(rewards))
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
                ax.plot(range(window-1, len(rewards)), smoothed,
                        color="navy", linewidth=1.8, label=f"MA-{window}")
        ax.set_title(
            f"Run {i}  lr={r['learning_rate']}  γ={r['gamma']}\n"
            f"batch={r['batch_size']}  buf={r['buffer_size']}",
            fontsize=8,
        )
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward",  fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    path = os.path.join(PLOT_DIR, "dqn_reward_curves.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Reward curves → {path}")

    # Summary bar chart
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    run_ids  = [r["run"] for r in results]
    means    = [r["mean_ep_reward"] for r in results]
    stds     = [r["std_ep_reward"]  for r in results]
    bars = ax2.bar(run_ids, means, yerr=stds, capsize=4,
                   color="steelblue", edgecolor="navy", linewidth=0.8)
    ax2.set_xlabel("Run ID", fontweight="bold")
    ax2.set_ylabel("Mean Episode Reward (last 20)", fontweight="bold")
    ax2.set_title("DQN — Hyperparameter Comparison Summary", fontweight="bold")
    ax2.set_xticks(run_ids)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    fig2.tight_layout()
    path2 = os.path.join(PLOT_DIR, "dqn_summary.png")
    fig2.savefig(path2, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Summary chart  → {path2}")

    # Loss curves (where available)
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for r in results:
        losses = r["losses"]
        if losses:
            window = min(50, len(losses))
            sm = np.convolve(losses, np.ones(window)/window, mode="valid")
            ax3.plot(sm, alpha=0.7, linewidth=0.9, label=f"Run {r['run']}")
    ax3.set_title("DQN — TD Loss Curves (smoothed)", fontweight="bold")
    ax3.set_xlabel("Gradient Step")
    ax3.set_ylabel("Loss")
    ax3.legend(fontsize=7, ncol=5)
    fig3.tight_layout()
    path3 = os.path.join(PLOT_DIR, "dqn_loss_curves.png")
    fig3.savefig(path3, dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Loss curves    → {path3}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DQN Hyperparameter Experiments")
    parser.add_argument("--timesteps", type=int,   default=100_000)
    parser.add_argument("--run_id",    type=int,   default=-1,
                        help="Train single run (0-9). -1 = all")
    parser.add_argument("--all",       action="store_true",
                        help="Train all 10 runs")
    parser.add_argument("--verbose",   type=int,   default=1)
    args = parser.parse_args()

    run_ids = list(range(len(DQN_CONFIGS))) if (args.all or args.run_id < 0) \
              else [args.run_id]

    print(f"\nDQN training: {len(run_ids)} run(s) × {args.timesteps:,} timesteps")

    results = []
    for idx in run_ids:
        r = train_dqn(idx, timesteps=args.timesteps, verbose=args.verbose)
        results.append(r)

    save_csv(results)
    plot_results(results)

    best = max(results, key=lambda x: x["mean_ep_reward"])
    print(f"\n  Best run: {best['run']} — mean_ep_reward={best['mean_ep_reward']}")


if __name__ == "__main__":
    main()
