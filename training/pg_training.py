"""
training/pg_training.py
=======================
Trains PPO agent on CropDiseaseEnv using Stable Baselines 3.
Each algorithm gets 10 hyperparameter configurations with results
logged to CSV and visualised.

Run:
    python -m training.pg_training --algo ppo [--all]
    python -m training.pg_training --algo a2c [--all]
    python -m training.pg_training --all       # trains both
"""

from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from environment.custom_env import CropDiseaseEnv

MODEL_DIR  = os.path.join("models", "pg")
RESULT_DIR = "results"
PLOT_DIR   = os.path.join("results", "plots")

os.makedirs(MODEL_DIR,  exist_ok=True)
os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PPO — 10 Hyperparameter configurations
# ─────────────────────────────────────────────────────────────────────────────

PPO_CONFIGS = [
    # Run 0 — baseline
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 1 — smaller steps, higher LR
    dict(learning_rate=1e-3, gamma=0.99, n_steps=512,  batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 2 — lower gamma (short horizon)
    dict(learning_rate=3e-4, gamma=0.95, n_steps=1024, batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 3 — tight clip range (conservative updates)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.1, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 4 — wide clip range (aggressive updates)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.3, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 5 — high entropy (exploration)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=1024, batch_size=64,
         n_epochs=10, ent_coef=0.05, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 6 — zero entropy (exploitation)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=1024, batch_size=64,
         n_epochs=10, ent_coef=0.00, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 7 — deeper network
    dict(learning_rate=3e-4, gamma=0.99, n_steps=2048, batch_size=128,
         n_epochs=5,  ent_coef=0.01, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[256, 256, 128]),

    # Run 8 — high vf_coef (stronger critic)
    dict(learning_rate=3e-4, gamma=0.99, n_steps=1024, batch_size=64,
         n_epochs=10, ent_coef=0.01, vf_coef=1.0,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),

    # Run 9 — low LR, many epochs per update
    dict(learning_rate=1e-4, gamma=0.99, n_steps=2048, batch_size=64,
         n_epochs=20, ent_coef=0.01, vf_coef=0.5,
         clip_range=0.2, max_grad_norm=0.5, net_arch=[128, 128]),
]


# ─────────────────────────────────────────────────────────────────────────────
# A2C — 10 Hyperparameter configurations
# ─────────────────────────────────────────────────────────────────────────────

A2C_CONFIGS = [
    # Run 0 — baseline
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 1 — longer rollouts
    dict(learning_rate=7e-4, gamma=0.99, n_steps=20,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 2 — Adam optimiser
    dict(learning_rate=3e-4, gamma=0.99, n_steps=5,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=False, net_arch=[128, 128]),

    # Run 3 — lower gamma
    dict(learning_rate=7e-4, gamma=0.90, n_steps=5,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 4 — high entropy
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5,
         ent_coef=0.05, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 5 — zero entropy
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5,
         ent_coef=0.00, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 6 — strong critic
    dict(learning_rate=7e-4, gamma=0.99, n_steps=5,
         ent_coef=0.01, vf_coef=1.0,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 7 — deeper network
    dict(learning_rate=5e-4, gamma=0.99, n_steps=10,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[256, 256]),

    # Run 8 — small LR
    dict(learning_rate=1e-4, gamma=0.99, n_steps=5,
         ent_coef=0.01, vf_coef=0.5,
         max_grad_norm=0.5, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 128]),

    # Run 9 — no grad clipping
    dict(learning_rate=7e-4, gamma=0.99, n_steps=15,
         ent_coef=0.02, vf_coef=0.5,
         max_grad_norm=10.0, rms_prop_eps=1e-5,
         use_rms_prop=True,  net_arch=[128, 64]),
]


# ── Callback ──────────────────────────────────────────────────────────────────

class MetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode_rewards: list[float] = []
        self.entropy_log: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        # SB3 stores entropy in logger
        try:
            ent = self.model.logger.name_to_value.get("train/entropy_loss", None)
            if ent is not None:
                self.entropy_log.append(float(ent))
        except Exception:
            pass
        return True


# ── Generic SB3 trainer ───────────────────────────────────────────────────────

def train_sb3(
    algo: str,
    config_idx: int,
    timesteps: int = 100_000,
    seed: int = 42,
    verbose: int = 1,
) -> dict:
    algo = algo.upper()
    cfg  = PPO_CONFIGS[config_idx] if algo == "PPO" else A2C_CONFIGS[config_idx]
    run_tag = f"{algo.lower()}_run_{config_idx}"

    print(f"\n{'='*60}")
    print(f"  {algo} Run {config_idx:>2} / 9")
    print(f"  lr={cfg['learning_rate']}, gamma={cfg['gamma']}, "
          f"n_steps={cfg['n_steps']}")
    print(f"  ent_coef={cfg['ent_coef']}, vf_coef={cfg['vf_coef']}, "
          f"net={cfg['net_arch']}")
    print(f"{'='*60}")

    def make_env():
        e = CropDiseaseEnv(grid_size=8, max_steps=250)
        return Monitor(e)

    n_envs   = 4 if algo == "A2C" else 1      # A2C benefits from parallel envs
    env      = make_vec_env(make_env, n_envs=n_envs, seed=seed)
    eval_env = make_vec_env(make_env, n_envs=1,      seed=seed + 200)

    callback     = MetricsCallback()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=RESULT_DIR,
        eval_freq=max(5000 // n_envs, 500),
        n_eval_episodes=10,
        deterministic=True,
        verbose=0,
    )

    policy_kwargs = {"net_arch": cfg["net_arch"]}

    if algo == "PPO":
        model = PPO(
            "MlpPolicy", env,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            clip_range=cfg["clip_range"],
            max_grad_norm=cfg["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            seed=seed, verbose=verbose, tensorboard_log=None,
        )
    else:  # A2C
        model = A2C(
            "MlpPolicy", env,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            ent_coef=cfg["ent_coef"],
            vf_coef=cfg["vf_coef"],
            max_grad_norm=cfg["max_grad_norm"],
            rms_prop_eps=cfg["rms_prop_eps"],
            use_rms_prop=cfg["use_rms_prop"],
            policy_kwargs=policy_kwargs,
            seed=seed, verbose=verbose, tensorboard_log=None,
        )

    t0 = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[callback, eval_callback],
        progress_bar=False,
    )
    elapsed = time.time() - t0

    model.save(os.path.join(MODEL_DIR, f"{run_tag}.zip"))

    ep_rewards = callback.episode_rewards
    mean_r = float(np.mean(ep_rewards[-20:])) if len(ep_rewards) >= 20 else float(np.mean(ep_rewards)) if ep_rewards else 0.0
    std_r  = float(np.std(ep_rewards[-20:]))  if len(ep_rewards) >= 20 else 0.0
    max_r  = float(np.max(ep_rewards))         if ep_rewards else 0.0

    print(f"  ✓  {algo} Run {config_idx} done in {elapsed:.1f}s | "
          f"mean_ep_reward={mean_r:.1f} ± {std_r:.1f}")

    env.close();  eval_env.close()

    row = {
        "run":             config_idx,
        "learning_rate":   cfg["learning_rate"],
        "gamma":           cfg["gamma"],
        "n_steps":         cfg["n_steps"],
        "ent_coef":        cfg["ent_coef"],
        "vf_coef":         cfg["vf_coef"],
        "net_arch":        str(cfg["net_arch"]),
        "mean_ep_reward":  round(mean_r, 2),
        "std_ep_reward":   round(std_r,  2),
        "max_ep_reward":   round(max_r,  2),
        "n_episodes":      len(ep_rewards),
        "train_time_s":    round(elapsed, 1),
        "episode_rewards": ep_rewards,
        "entropy_log":     callback.entropy_log,
    }

    if algo == "PPO":
        row["batch_size"]  = cfg["batch_size"]
        row["n_epochs"]    = cfg["n_epochs"]
        row["clip_range"]  = cfg["clip_range"]
    else:
        row["use_rms_prop"] = cfg["use_rms_prop"]

    return row


# ── Save / plot helpers ───────────────────────────────────────────────────────

def save_csv(results: list[dict], algo: str):
    path = os.path.join(RESULT_DIR, f"{algo.lower()}_results.csv")
    skip = {"episode_rewards", "entropy_log"}
    keys = [k for k in results[0].keys() if k not in skip]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in keys})
    print(f"  Results CSV → {path}")


def plot_results(results: list[dict], algo: str):
    n = len(results)
    cols = min(5, n);  rows = (n + cols - 1) // cols
    color_map = {"PPO": "teal", "A2C": "darkorange"}
    col = color_map.get(algo.upper(), "purple")

    # Reward subplots
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows),
                              constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle(f"{algo.upper()} — Episode Reward per Run",
                 fontsize=14, fontweight="bold")

    for i, r in enumerate(results):
        ax  = axes[i]
        rew = r["episode_rewards"]
        if rew:
            ax.plot(rew, alpha=0.3, color=col, linewidth=0.7)
            w = min(20, len(rew))
            if len(rew) >= w:
                sm = np.convolve(rew, np.ones(w)/w, mode="valid")
                ax.plot(range(w-1, len(rew)), sm,
                        color="black", linewidth=1.8)
        ax.set_title(
            f"Run {i}  lr={r['learning_rate']}  γ={r['gamma']}\n"
            f"ent={r['ent_coef']}  vf={r['vf_coef']}",
            fontsize=8,
        )
        ax.set_xlabel("Episode", fontsize=7)
        ax.set_ylabel("Reward",  fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes[n:]:
        ax.set_visible(False)

    p = os.path.join(PLOT_DIR, f"{algo.lower()}_reward_curves.png")
    fig.savefig(p, dpi=130, bbox_inches="tight");  plt.close(fig)
    print(f"  Reward curves → {p}")

    # Entropy subplot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for r in results:
        ent = r["entropy_log"]
        if len(ent) >= 10:
            w  = min(20, len(ent))
            sm = np.convolve(ent, np.ones(w)/w, mode="valid")
            ax2.plot(sm, alpha=0.7, linewidth=0.9, label=f"Run {r['run']}")
    ax2.set_title(f"{algo.upper()} — Entropy Loss Curves", fontweight="bold")
    ax2.set_xlabel("Update step");  ax2.set_ylabel("Entropy loss")
    ax2.legend(fontsize=7, ncol=5)
    fig2.tight_layout()
    p2 = os.path.join(PLOT_DIR, f"{algo.lower()}_entropy.png")
    fig2.savefig(p2, dpi=130, bbox_inches="tight");  plt.close(fig2)
    print(f"  Entropy curves → {p2}")

    # Summary bar
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    run_ids = [r["run"] for r in results]
    means   = [r["mean_ep_reward"] for r in results]
    stds    = [r["std_ep_reward"]  for r in results]
    bars = ax3.bar(run_ids, means, yerr=stds, capsize=4,
                   color=col, edgecolor="black", linewidth=0.7)
    ax3.set_title(f"{algo.upper()} — Hyperparameter Comparison",
                  fontweight="bold")
    ax3.set_xlabel("Run ID");  ax3.set_ylabel("Mean Episode Reward (last 20)")
    ax3.set_xticks(run_ids)
    for bar, val in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    fig3.tight_layout()
    p3 = os.path.join(PLOT_DIR, f"{algo.lower()}_summary.png")
    fig3.savefig(p3, dpi=130, bbox_inches="tight");  plt.close(fig3)
    print(f"  Summary chart  → {p3}")


# ── All-algorithm comparison ──────────────────────────────────────────────────

def plot_all_comparison(ppo_results, a2c_results):
    """Combined subplot comparing all four algorithms' best runs."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("All Algorithms — Best Run Comparison", fontsize=14, fontweight="bold")

    pairs = [
        (ppo_results, "PPO", "teal",      axes[0, 0]),
        (a2c_results, "A2C", "darkorange", axes[0, 1]),
    ]
    for results, algo, col, ax in pairs:
        best = max(results, key=lambda x: x["mean_ep_reward"])
        rew  = best["episode_rewards"]
        ax.plot(rew, alpha=0.3, color=col, linewidth=0.7)
        w = min(30, len(rew))
        if len(rew) >= w:
            sm = np.convolve(rew, np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(rew)), sm, color="black", linewidth=2.0,
                    label=f"MA-{w}")
        ax.set_title(f"{algo} Best (Run {best['run']})", fontweight="bold")
        ax.set_xlabel("Episode");  ax.set_ylabel("Reward")
        ax.legend(fontsize=9)

    # Comparison bar
    ax_bar = axes[1, 0]
    algos  = ["PPO", "A2C"]
    best_means = [
        max(ppo_results, key=lambda x: x["mean_ep_reward"])["mean_ep_reward"],
        max(a2c_results, key=lambda x: x["mean_ep_reward"])["mean_ep_reward"],
    ]
    best_stds  = [
        max(ppo_results, key=lambda x: x["mean_ep_reward"])["std_ep_reward"],
        max(a2c_results, key=lambda x: x["mean_ep_reward"])["std_ep_reward"],
    ]
    cols_ = ["teal", "darkorange"]
    ax_bar.bar(algos, best_means, yerr=best_stds, capsize=6,
               color=cols_, edgecolor="black", linewidth=0.8)
    ax_bar.set_title("Best Run per Algorithm", fontweight="bold")
    ax_bar.set_ylabel("Mean Episode Reward")
    for i, (algo, val) in enumerate(zip(algos, best_means)):
        ax_bar.text(i, val + 0.5, f"{val:.1f}", ha="center",
                    va="bottom", fontweight="bold")

    axes[1, 1].set_visible(False)
    fig.tight_layout()
    p = os.path.join(PLOT_DIR, "pg_all_comparison.png")
    fig.savefig(p, dpi=130, bbox_inches="tight");  plt.close(fig)
    print(f"  All-algo comparison → {p}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PPO / A2C Hyperparameter Experiments")
    parser.add_argument("--algo",      type=str, default="ppo",
                        choices=["ppo", "a2c", "both"])
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--run_id",    type=int, default=-1,
                        help="Single run index (0-9). -1 = all")
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--verbose",   type=int, default=1)
    args = parser.parse_args()

    algo_list = ["ppo", "a2c"] if args.algo == "both" else [args.algo]
    run_ids   = list(range(10)) if (args.all or args.run_id < 0) \
                else [args.run_id]

    all_results = {}
    for algo in algo_list:
        print(f"\n{'#'*60}")
        print(f"  Training {algo.upper()} — {len(run_ids)} run(s)")
        print(f"{'#'*60}")
        results = []
        for idx in run_ids:
            r = train_sb3(algo, idx, timesteps=args.timesteps,
                          verbose=args.verbose)
            results.append(r)
        save_csv(results, algo)
        plot_results(results, algo)
        all_results[algo] = results
        best = max(results, key=lambda x: x["mean_ep_reward"])
        print(f"\n  Best {algo.upper()} run: {best['run']} — "
              f"mean_ep_reward={best['mean_ep_reward']}")

    if "ppo" in all_results and "a2c" in all_results:
        plot_all_comparison(all_results["ppo"], all_results["a2c"])


if __name__ == "__main__":
    main()
