# CropCare RL: AI-Powered Crop Disease Management

**Student:** Marie Elyse  
**Course:** Machine Learning Techniques II — Reinforcement Learning  
**Institution:** African Leadership University (ALU)  
**Video Demo:** [Watch on YouTube](#)  
**Mission:** Protecting smallholder farmers in Rwanda from crop disease through autonomous AI-driven farm management

\---

## Project Overview

This project implements and compares three reinforcement learning algorithms to autonomously manage crop disease detection and treatment on a simulated farm. The AI agent learns to patrol an 8×8 farm grid, inspect plant cells to reveal hidden disease levels, and apply targeted treatment — maximising farm health while conserving limited treatment resources.

**Mission Context:** Rwanda's smallholder farmers lose up to 40% of crop yields annually due to undetected diseases. Early detection and targeted treatment can dramatically reduce these losses. This project develops an AI agent that mimics how a precision agriculture drone would operate — autonomously scanning fields and treating only what needs treatment.

\---

## Project Structure

```
Marie\_Elyse\_Uyiringiye\_rl\_summative/
├── environment/
│   ├── \_\_init\_\_.py
│   ├── custom\_env.py            # Custom Gymnasium CropDiseaseEnv
│   └── rendering.py             # Pygame 2D visualisation system
├── training/
│   ├── \_\_init\_\_.py
│   ├── dqn\_training.py          # DQN — 10 hyperparameter configs
│   ├── pg\_training.py           # PPO — 10 hyperparameter configs
│   └── reinforce.py             # Custom REINFORCE (PyTorch)
├── models/
│   ├── dqn/                     # Saved DQN models (.zip)
│   └── pg/                      # Saved PPO and REINFORCE models
├── results/
│   ├── plots/                   # Learning curves and comparison charts
│   ├── videos/                  # Simulation recordings
│   ├── dqn\_results.csv
│   ├── ppo\_results.csv
│   └── reinforce\_results.csv
├── main.py                      # Entry point — run best agent
├── random\_demo.py               # Random agent demo (no training)
├── requirements.txt
└── README.md
```

\---

## Environment Details

### Observation Space (196-dimensional)

|Component|Size|Description|
|-|-|-|
|Agent position|2|Row and col normalised to \[0, 1]|
|Visible disease per cell|64|Revealed after inspection, 0 if unknown|
|Inspection mask|64|1 if inspected, 0 otherwise|
|Treatment mask|64|1 if treated, 0 otherwise|
|Doses remaining|1|Normalised remaining treatment doses|
|Steps remaining|1|Normalised remaining time steps|
|**Total**|**196**|**Box(196,) float32**|

### Action Space (Discrete 8)

|ID|Action|Effect|
|-|-|-|
|0|Move North|Move one cell north, auto-inspect new cell|
|1|Move South|Move one cell south, auto-inspect new cell|
|2|Move West|Move one cell west, auto-inspect new cell|
|3|Move East|Move one cell east, auto-inspect new cell|
|4|Inspect|Reveal disease level of current cell|
|5|Light Treat|1 dose — cures Mild and Moderate disease|
|6|Heavy Treat|2 doses — cures all disease levels|
|7|Mark Healthy|Skip cell, no treatment applied|

### Disease Levels

|Level|Name|Colour|
|-|-|-|
|0|Healthy|Green|
|1|Mild|Yellow|
|2|Moderate|Orange|
|3|Severe|Red|
|4|Dead|Dark Grey|

### Reward Structure

|Event|Reward|
|-|-|
|New cell inspected|+1.0|
|Disease found (severity d)|+2d|
|Correct light treatment|+10.0|
|Correct heavy treatment|+15.0|
|Correct healthy skip|+5.0|
|Treating healthy plant|-3.0 to -5.0|
|Missing disease (severity d)|-8d|
|Per step cost|-0.3|
|Completion bonus|+30 + 20 x efficiency|

### Episode Termination

* **Terminated:** All 64 cells inspected and decided upon
* **Truncated:** Maximum 250 steps reached

\---

## Quick Start

### Prerequisites

* Python 3.8+
* pip

### Installation

```bash
# Clone the repository
git clone https://github.com/elyse003/Marie\_Elyse\_Uyiringiye\_rl\_summative.git
cd Marie\_Elyse\_Uyiringiye\_rl\_summative

# Install dependencies
pip install -r requirements.txt
```

### Run Pre-Trained Models

```bash
# Run best performing agent (PPO)
python main.py --algo ppo --run 0 --episodes 3

# Run DQN agent
python main.py --algo dqn --run 9 --episodes 3

# Run REINFORCE agent
python main.py --algo reinforce --run 0 --episodes 3

# Random agent demo (no model needed)
python random\_demo.py --episodes 1
```

### Train from Scratch

```bash
# Train DQN — 10 configurations
python -m training.dqn\_training --all --timesteps 300000

# Train PPO — 10 configurations
python -m training.pg\_training --algo ppo --all --timesteps 300000

# Train REINFORCE — 10 configurations
python -m training.reinforce --all --episodes 2000
```

\---

## Algorithms Implemented

### 1\. DQN — Value-Based

* **Architecture:** MLP \[128, 128] hidden layers
* **Key Features:** Experience replay buffer, target network, epsilon-greedy exploration
* **Best Run:** Run 9 — lr=5e-4, gamma=0.99, buffer=75k, net=\[512]
* **Performance:** 13.0 / 64 cells inspected on average
* **Finding:** Converged to reward exploit — treating start cell repeatedly instead of exploring

### 2\. PPO — Policy Gradient

* **Architecture:** Shared \[128, 128] MLP with separate policy and value heads
* **Key Features:** Clipped surrogate objective, entropy bonus, 4 parallel environments
* **Best Run:** Run 0 — lr=3e-4, gamma=0.99, n\_steps=2048, ent\_coef=0.01
* **Performance:** 38.7 / 64 cells inspected (57.8% farm coverage)
* **Finding:** Best exploration due to entropy regularisation forcing diverse action selection

### 3\. REINFORCE — Policy Gradient (Custom PyTorch)

* **Architecture:** \[128, 128] MLP with softmax output
* **Key Features:** Monte Carlo returns, return normalisation, entropy bonus, gradient clipping
* **Best Run:** Run 0 — lr=1e-3, gamma=0.99, normalise=True, entropy=0.01
* **Finding:** High variance typical of Monte Carlo methods, more stable with return normalisation

\---

## Hyperparameter Tuning

10 configurations per algorithm = **30 total training runs**

Parameters tuned across all algorithms:

* Learning rates: 1e-4 to 2e-3
* Network architectures: \[64, 64] to \[512]
* Discount factors (gamma): 0.90 to 0.99
* Entropy coefficients: 0.00 to 0.05
* Exploration fractions (DQN): 0.10 to 0.30
* Clip ranges (PPO): 0.1 to 0.3
* n\_steps (PPO): 512 to 2048
* Batch sizes: 32 to 256

All results saved to `results/{algo}\_results.csv`

\---

## Performance Results

|Algorithm|Best Run|Avg Inspected|Avg Reward|
|-|-|-|-|
|**PPO**|**Run 0**|**38.7 / 64**|**-192.3**|
|DQN|Run 9|13.0 / 64|-250.6|
|REINFORCE|Run 0|—|—|

**Key Finding:** PPO outperformed DQN because its entropy regularization explicitly
incentivizes exploration. DQN converged to a suboptimal exploit — treating the
starting cell repeatedly for immediate reward — rather than learning the full
inspection-treatment pipeline across the grid.

\---

## Visualisation

The Pygame rendering system features:

* **Farm Grid:** 8x8 color-coded disease grid from green (healthy) to dark grey (dead)
* **Agent:** White circle with directional arrow showing current position
* **HUD Panel:** Real-time episode stats — total reward, sick cells, inspected count, doses remaining
* **Resource Bars:** Animated dose and time progress bars
* **Status Bar:** Inspection and treatment coverage percentages at the bottom

\---

## Mission Alignment: CropCare Rwanda

**The Problem**  
Rwandan smallholder farmers lose 20-40% of their yields to undetected crop diseases.
Manual inspection is slow, expensive, and often too late to prevent spread.

**Our Solution**  
An AI agent autonomously patrols and inspects every plant cell, treating only
what needs treating, and how to conserve limited treatment resources.

**Impact**

* Reduces crop losses through early, systematic disease detection
* Conserves treatment resources through intelligent prioritization
* Makes precision agriculture accessible without expert knowledge
* Provides a foundation for integration with drone or mobile app hardware

\---

## Dependencies

```
gymnasium>=0.29.0
stable-baselines3>=2.2.0
torch>=2.0.0
pygame>=2.5.0
matplotlib>=3.7.0
numpy>=1.24.0
imageio>=2.31.0
imageio-ffmpeg>=0.4.9
Pillow>=10.0.0
pandas>=2.0.0
```

\---

## Video Demo

\[Insert YouTube link here]

## GitHub Repository

\[https://github.com/elyse003/Marie\_Elyse\_Uyiringiye\_rl\_summative.git]

\---

**Student:** Marie Elyse | African Leadership University (ALU)

