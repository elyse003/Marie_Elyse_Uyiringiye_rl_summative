# CropCare RL Summative
### Reinforcement Learning for Crop Disease Management
**Author:** Cedric — African Leadership University

---

## Problem Statement

Smallholder farmers in Rwanda lose 20–40% of yields to undetected crop diseases. This project builds an RL agent that autonomously **patrols a farm grid**, **inspects plant cells** for hidden disease, and **applies targeted treatment** — mimicking how a precision agriculture drone or robot would operate.

The agent must balance: thorough inspection, efficient resource usage (treatment doses), and correct diagnosis (no over- or under-treatment).

---

## Environment: `CropDiseaseEnv`

| Property | Value |
|---|---|
| Grid | 8 × 8 farm cells |
| Disease states | 0=Healthy, 1=Mild, 2=Moderate, 3=Severe, 4=Dead |
| Disease hidden? | Yes — agent must INSPECT to reveal |
| Disease spreads | Stochastically to neighbours (4% chance/step) |
| Observation space | Box(196,) — position + disease visibility + resources |
| Action space | Discrete(8) |

### Actions

| ID | Action | Description |
|---|---|---|
| 0 | Move North | Navigate grid |
| 1 | Move South | Navigate grid |
| 2 | Move West | Navigate grid |
| 3 | Move East | Navigate grid |
| 4 | Inspect | Reveal true disease level of current cell |
| 5 | Light Treat | 1 dose — cures Mild/Moderate |
| 6 | Heavy Treat | 2 doses — cures all disease levels |
| 7 | Mark Healthy | Skip cell (commit to no treatment) |

### Rewards

| Event | Reward |
|---|---|
| New inspection | +1.0 |
| Found disease (severity d) | +2d |
| Correct light treatment | +10 |
| Correct heavy treatment | +15 |
| Correct healthy skip | +5 |
| Treating healthy plant (light) | -3 |
| Treating healthy plant (heavy) | -5 |
| Missing disease (mark healthy) | -8d |
| Per step | -0.5 |
| Completion bonus | +30 + 20×efficiency |

### Terminal Conditions
- **Terminated:** All cells inspected/decided
- **Truncated:** `max_steps=250` reached

---

## Algorithms

| Algorithm | Library | Script |
|---|---|---|
| DQN | Stable Baselines 3 | `training/dqn_training.py` |
| PPO | Stable Baselines 3 | `training/pg_training.py` |
| A2C | Stable Baselines 3 | `training/pg_training.py` |
| REINFORCE | Custom PyTorch | `training/reinforce.py` |

Each algorithm has **10 hyperparameter configurations**.

---

## Project Structure

```
cedric_rl_summative/
├── environment/
│   ├── custom_env.py        # CropDiseaseEnv (Gymnasium)
│   └── rendering.py         # Pygame visualisation
├── training/
│   ├── dqn_training.py      # DQN — 10 configs
│   ├── pg_training.py       # PPO + A2C — 10 configs each
│   └── reinforce.py         # REINFORCE — 10 configs (custom)
├── models/
│   ├── dqn/                 # Saved DQN models
│   └── pg/                  # Saved PPO, A2C, REINFORCE models
├── results/
│   ├── plots/               # Learning curves, entropy, summaries
│   └── videos/              # Simulation recordings
├── main.py                  # Run best agent simulation
├── random_demo.py           # Random agent (no model)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Watch random agent (no training needed)
python random_demo.py --episodes 1

# 2. Train all algorithms (adjust timesteps for speed)
python -m training.dqn_training --all --timesteps 100000
python -m training.pg_training  --algo ppo --all --timesteps 100000
python -m training.pg_training  --algo a2c --all --timesteps 100000
python -m training.reinforce    --all --episodes 1000

# 3. Run best agent simulation
python main.py --algo ppo --episodes 3 --record
```

### Kaggle Notebook (recommended)

```python
# Cell 1 — Setup
import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt", "-q"])

# Cell 2 — Train DQN
%run training/dqn_training.py --all --timesteps 100000

# Cell 3 — Train PPO + A2C
%run training/pg_training.py --algo both --all --timesteps 100000

# Cell 4 — Train REINFORCE
%run training/reinforce.py --all --episodes 1000

# Cell 5 — Run simulation
%run main.py --algo ppo --episodes 3 --record
```

---

## Results

After training, find results in:
- `results/dqn_results.csv`, `results/ppo_results.csv`, etc.
- `results/plots/` — all learning curve images
- `results/videos/` — simulation recordings

---

## Environment Diagram

```
┌──────────── 8 × 8 Farm Grid ────────────┐
│  G  G  M  G  G  S  G  G   ← True state │
│  G  M  G  G  D  G  G  M   (hidden from  │
│  G  G  G  M  G  G  S  G    agent until  │
│  M  G  G  G  G  M  G  G    inspected)  │
│  G  G  S  G  G  G  D  G                │
│  G  M  G  G  M  G  G  G                │
│  G  G  G  D  G  G  G  M                │
│  G  G  G  G  G  S  G  G   → 64 cells  │
└─────────────────────────────────────────┘
  G=Healthy  M=Mild  S=Severe  D=Dead

              ↓ Agent inspects cell
┌────────────── Agent View ───────────────┐
│  ?  ?  M  ?  ?  ?  ?  ?   ← Partial   │
│  ?  ?  ?  ?  ?  ?  ?  ?    knowledge  │
│  ?  ?  ?  ?  ?  ?  ?  ?               │
│  ? [A] ?  ?  ?  ?  ?  ?   A = Agent  │
│  ...                                    │
└─────────────────────────────────────────┘

Actions: Move (4) → Inspect → Treat/Skip
```
