"""
main.py — CropCare RL Entry Point
==================================
Loads the best-performing trained model and runs a live simulation
with full GUI rendering and terminal verbose output.

Usage
-----
    python main.py                          # auto-select best model
    python main.py --algo dqn --run 3       # specific DQN run
    python main.py --algo ppo --run 0
    python main.py --algo reinforce --run 1
    python main.py --episodes 5 --record    # record to video
"""

from __future__ import annotations

import os
import sys
import argparse
import time
import glob
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, os.path.dirname(__file__))
from environment.custom_env import CropDiseaseEnv, ACTION_LABELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(algo: str, run_id: int | None):
    """Load and return (model, algo_name, run_id) for the given algo."""
    algo = algo.lower()

    if algo == "reinforce":
        return _load_reinforce(run_id)

    from stable_baselines3 import DQN, PPO
    cls_map = {"dqn": DQN, "ppo": PPO}
    model_dir = os.path.join("models", "dqn" if algo == "dqn" else "pg")

    if run_id is not None:
        path = os.path.join(model_dir, f"{algo}_run_{run_id}.zip")
    else:
        # Try to find best saved model
        candidates = glob.glob(os.path.join(model_dir, f"{algo}_run_*.zip"))
        if not candidates:
            # Fall back to SB3 EvalCallback best_model
            candidates = glob.glob(os.path.join(model_dir, "best_model.zip"))
        if not candidates:
            raise FileNotFoundError(
                f"No {algo.upper()} model found in {model_dir}/\n"
                f"Run training first: python -m training.{'dqn_training' if algo=='dqn' else 'pg_training'} --all"
            )
        path = sorted(candidates)[-1]   # last = most recent
        run_id = _extract_run_id(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")

    print(f"  Loading {algo.upper()} model from: {path}")
    model = cls_map[algo].load(path)
    return model, algo, run_id


def _load_reinforce(run_id: int | None):
    import torch
    from training.reinforce import PolicyNetwork, REINFORCE_CONFIGS

    model_dir = os.path.join("models", "pg")
    if run_id is not None:
        path = os.path.join(model_dir, f"reinforce_run_{run_id}.pt")
    else:
        candidates = glob.glob(os.path.join(model_dir, "reinforce_run_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No REINFORCE model found. Run: python -m training.reinforce --all"
            )
        path = sorted(candidates)[-1]
        run_id = _extract_run_id(path)

    print(f"  Loading REINFORCE model from: {path}")
    cfg    = REINFORCE_CONFIGS[run_id or 0]
    env_   = CropDiseaseEnv()
    policy = PolicyNetwork(
        obs_dim=env_.observation_space.shape[0],
        act_dim=env_.action_space.n,
        hidden_sizes=cfg["hidden_sizes"],
    )
    env_.close()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.load_state_dict(torch.load(path, map_location=device))
    policy.eval()

    class REINFORCEWrapper:
        """Wraps our policy so it mimics SB3's predict() interface."""
        def __init__(self, p):
            self.p = p
        def predict(self, obs, deterministic=True):
            import torch
            x      = torch.FloatTensor(obs).unsqueeze(0).to(device)
            probs  = self.p(x).squeeze(0)
            action = probs.argmax().item() if deterministic else \
                     torch.distributions.Categorical(probs).sample().item()
            return np.array([action]), None

    return REINFORCEWrapper(policy), "reinforce", run_id


def _extract_run_id(path: str) -> int:
    import re
    m = re.search(r"_(\d+)\.(zip|pt)$", path)
    return int(m.group(1)) if m else 0


# ── Run simulation ────────────────────────────────────────────────────────────

def run_simulation(
    model,
    algo: str,
    run_id: int,
    n_episodes: int = 3,
    record: bool = False,
    render_mode: str = "rgb_array",
    fps: int = 10,
    seed: int = 0,
):
    os.makedirs("results/videos", exist_ok=True)

    env = CropDiseaseEnv(
        grid_size=8,
        max_steps=250,
        render_mode=render_mode,
    )

    print("\n" + "=" * 65)
    print(f"  CropCare RL — Simulation")
    print(f"  Algorithm : {algo.upper()}  (run {run_id})")
    print(f"  Episodes  : {n_episodes}")
    print(f"  Obs dim   : {env.observation_space.shape[0]}")
    print(f"  Actions   : {env.action_space.n}")
    print("=" * 65)
    print()
    print("  Problem:")
    print("    Smallholder farms in Rwanda suffer significant crop")
    print("    losses from undetected diseases. The agent patrols an")
    print("    8×8 farm grid, INSPECTS cells to reveal hidden disease")
    print("    levels, then decides: light treat, heavy treat, or skip.")
    print()
    print("  Reward structure:")
    print("    +1.0  per new inspection")
    print("    +2*d  discovering disease severity d")
    print("    +10   correct light treatment (mild/moderate)")
    print("    +15   correct heavy treatment (severe/dead)")
    print("    +5    correctly skip healthy cell")
    print("    -0.5  per step  (efficiency pressure)")
    print("    -3/-5 treating healthy plant (waste)")
    print("    +30+20*eff  completion bonus")
    print()
    print("  Agent objective:")
    print("    Maximise total farm health score by efficiently")
    print("    inspecting all cells and applying correct treatments.")
    print("=" * 65)

    all_rewards = []
    all_steps   = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        frames     = []
        ep_reward  = 0.0
        step       = 0
        action_counts = {k: 0 for k in ACTION_LABELS}

        print(f"\n┌── Episode {ep+1}/{n_episodes}  "
              f"(diseased cells: {info['total_diseased']}/{info['total_cells']}) ──")

        t_start = time.time()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step      += 1
            action_counts[action] = action_counts.get(action, 0) + 1

            frame = env.render()
            if frame is not None and record:
                frames.append(frame)

            # Verbose terminal output every 25 steps
            if step % 25 == 0:
                print(f"│  Step {step:>4}  pos={info['agent_pos']}  "
                      f"action={ACTION_LABELS[action]:<14}  "
                      f"r={reward:>+7.2f}  total={ep_reward:>+8.2f}  "
                      f"insp={info['cells_inspected']}/{info['total_cells']}  "
                      f"trt={info['cells_treated']}")

            if terminated or truncated:
                elapsed = time.time() - t_start
                reason  = "✓ COMPLETE" if terminated else " TIME LIMIT"
                print(f"│")
                print(f"│  {reason}  in {step} steps ({elapsed:.1f}s)")
                print(f"│  Total reward      : {ep_reward:>+.2f}")
                print(f"│  Inspected         : {info['cells_inspected']}/{info['total_cells']}  "
                      f"({info['completion_pct']:.1f}%)")
                print(f"│  Treated           : {info['cells_treated']}")
                print(f"│  Correct treats    : {info['correct_treats']}")
                print(f"│  Missed diseases   : {info['missed_diseases']}")
                print(f"│  Doses remaining   : {info['doses_remaining']}")
                top_actions = sorted(action_counts.items(),
                                     key=lambda x: -x[1])[:3]
                print(f"│  Top actions       : "
                      + ", ".join(f"{ACTION_LABELS[a]}×{c}"
                                  for a, c in top_actions))
                print(f"└{'─'*60}")
                break

        all_rewards.append(ep_reward)
        all_steps.append(step)

        if frames and record:
            _save_video(frames, algo, run_id, ep, fps)

    env.close()

    print(f"\n  ── Summary across {n_episodes} episode(s) ──")
    print(f"  Mean reward : {np.mean(all_rewards):>+.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean steps  : {np.mean(all_steps):.1f}")
    print(f"  Best episode: {max(all_rewards):>+.2f}")
    print()


def _save_video(frames, algo, run_id, ep, fps):
    path = f"results/videos/{algo}_run{run_id}_ep{ep+1}.mp4"
    try:
        import imageio
        writer = imageio.get_writer(path, fps=fps, codec="libx264",
                                    quality=7, macro_block_size=None)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"\n  Video saved → {path}")
        return
    except Exception as e:
        pass

    gif_path = path.replace(".mp4", ".gif")
    try:
        import imageio
        step = max(1, len(frames) // 120)
        imageio.mimsave(gif_path, frames[::step], fps=fps)
        print(f"\n  GIF saved → {gif_path}")
    except Exception as e:
        print(f"\n  Could not save video: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CropCare RL Simulation")
    parser.add_argument("--algo",      type=str, default="dqn",
                        choices=["dqn", "ppo", "a2c", "reinforce"])
    parser.add_argument("--run",       type=int, default=None,
                        help="Model run index (default: auto-select)")
    parser.add_argument("--episodes",  type=int, default=3)
    parser.add_argument("--record",    action="store_true",
                        help="Record simulation to video")
    parser.add_argument("--seed",      type=int, default=0)
    args = parser.parse_args()

    model, algo, run_id = load_model(args.algo, args.run)

    run_simulation(
        model,
        algo=algo,
        run_id=run_id,
        n_episodes=args.episodes,
        record=args.record,
        render_mode="rgb_array" if args.record else "rgb_array",
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
