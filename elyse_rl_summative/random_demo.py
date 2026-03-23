"""
random_demo.py
==============
Demonstrates the CropCare environment with a RANDOM agent (no training).
Records a video and saves sample frames.

Run:
    python random_demo.py [--steps 200] [--episodes 3] [--output results/videos]
"""

import os
import sys
import argparse
import numpy as np

# ── headless pygame BEFORE any import that touches pygame ────────────────────
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


def main():
    parser = argparse.ArgumentParser(description="Random agent demo")
    parser.add_argument("--steps",    type=int, default=200,           help="Max steps per episode")
    parser.add_argument("--episodes", type=int, default=1,             help="Number of episodes to record")
    parser.add_argument("--output",   type=str, default="results/videos", help="Output directory")
    parser.add_argument("--fps",      type=int, default=8,             help="Video FPS")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    from environment.custom_env import CropDiseaseEnv, ACTION_LABELS

    env = CropDiseaseEnv(
        grid_size=8,
        max_steps=args.steps,
        render_mode="rgb_array",
    )

    print("=" * 60)
    print("  CropCare RL — Random Agent Demo")
    print("=" * 60)
    print(f"  Observation space : {env.observation_space.shape}")
    print(f"  Action space      : {env.action_space.n} discrete actions")
    print(f"  Actions           : {list(ACTION_LABELS.values())}")
    print("=" * 60)

    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep * 42)
        frames = []
        total_reward = 0.0
        step = 0

        print(f"\nEpisode {ep+1} starting …  "
              f"(diseased cells at reset: {info['total_diseased']})")

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            frame = env.render()
            if frame is not None:
                frames.append(frame)

            if step % 50 == 0:
                total_c = env.grid_size ** 2
                print(f"  Step {step:>4} | action={ACTION_LABELS[action]:<14} | "
                      f"reward={reward:>+7.2f} | total={total_reward:>+8.2f} | "
                      f"inspected={info['cells_inspected']}/{total_c}")

            if terminated or truncated:
                reason = "DONE (all cells decided)" if terminated else "TRUNCATED (time limit)"
                print(f"\n  Episode {ep+1} finished — {reason}")
                print(f"  Total reward    : {total_reward:.2f}")
                print(f"  Steps taken     : {step}")
                print(f"  Cells inspected : {info['cells_inspected']}/{info['total_cells']}")
                print(f"  Cells treated   : {info['cells_treated']}")
                print(f"  Correct treats  : {info['correct_treats']}")
                print(f"  Missed diseases : {info['missed_diseases']}")
                print(f"  Doses remaining : {info['doses_remaining']}")
                break

        if frames:
            _save_video(frames, args.output, ep, args.fps)

    env.close()
    print("\nDemo complete. Files written to:", args.output)


def _save_video(frames, out_dir, episode, fps):
    """Save frames to MP4 using imageio (preferred) or fallback to GIF."""
    path_mp4 = os.path.join(out_dir, f"random_agent_ep{episode+1}.mp4")
    path_gif = os.path.join(out_dir, f"random_agent_ep{episode+1}.gif")

    try:
        import imageio
        writer = imageio.get_writer(path_mp4, fps=fps, codec="libx264",
                                    quality=7, macro_block_size=None)
        for f in frames:
            writer.append_data(f)
        writer.close()
        print(f"\n  Video saved → {path_mp4}  ({len(frames)} frames @ {fps} fps)")
        return
    except Exception as e:
        print(f"  [MP4 failed: {e}] — trying GIF …")

    try:
        import imageio
        # Downsample for GIF to keep file size reasonable
        step = max(1, len(frames) // 120)
        imageio.mimsave(path_gif, frames[::step], fps=fps)
        print(f"  GIF saved → {path_gif}")
        return
    except Exception as e:
        print(f"  [GIF failed too: {e}] — saving raw frames …")

    # Last resort: save PNG frames
    frame_dir = os.path.join(out_dir, f"frames_ep{episode+1}")
    os.makedirs(frame_dir, exist_ok=True)
    from PIL import Image
    step = max(1, len(frames) // 30)
    for i, f in enumerate(frames[::step]):
        Image.fromarray(f).save(os.path.join(frame_dir, f"frame_{i:04d}.png"))
    print(f"  PNG frames saved → {frame_dir}/")


if __name__ == "__main__":
    main()
