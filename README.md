# \# CropCare RL — AI-Powered Crop Disease Management

# 

# \*\*Student:\*\* Marie Elyse Uyiringiye | \*\*Institution:\*\* African Leadership University (ALU)  

# \*\*Course:\*\* Machine Learning Techniques II — Reinforcement Learning  

# \*\*Video Demo:\*\* \[Watch on YouTube](#) | \*\*GitHub:\*\* \[Marie\_Elyse\_Uyiringiye\_rl\_summative](#)  

# \*\*Mission:\*\* Protecting smallholder farmers in Rwanda from crop disease through autonomous AI-driven farm management

# 

# \---

# 

# \## Project Overview

# 

# This project implements and compares three reinforcement learning algorithms — DQN, PPO, and REINFORCE — to autonomously manage crop disease on a simulated 8×8 farm grid. The agent inspects plant cells to reveal hidden disease levels and applies targeted treatment while conserving limited resources. Aligned with the CropCare mission of reducing yield losses for smallholder farmers in Rwanda.

# 

# \---

# 

# \## Project Structure

# 

# &#x20;   Marie\_Elyse\_Uyiringiye\_rl\_summative/

# &#x20;   ├── environment/

# &#x20;   │   ├── custom\_env.py       # Gymnasium CropDiseaseEnv

# &#x20;   │   └── rendering.py        # Pygame visualisation

# &#x20;   ├── training/

# &#x20;   │   ├── dqn\_training.py     # DQN — 10 configs

# &#x20;   │   ├── pg\_training.py      # PPO — 10 configs

# &#x20;   │   └── reinforce.py        # REINFORCE — custom PyTorch

# &#x20;   ├── models/

# &#x20;   │   ├── dqn/                # Saved DQN models (.zip)

# &#x20;   │   └── pg/                 # Saved PPO + REINFORCE models

# &#x20;   ├── results/

# &#x20;   │   ├── plots/              # Learning curves, comparison charts

# &#x20;   │   ├── videos/             # Simulation recordings

# &#x20;   │   ├── dqn\_results.csv

# &#x20;   │   ├── ppo\_results.csv

# &#x20;   │   └── reinforce\_results.csv

# &#x20;   ├── main.py

# &#x20;   ├── random\_demo.py

# &#x20;   ├── requirements.txt

# &#x20;   └── README.md

# 

# \---

# 

# \## Environment

# 

# \*\*Observation space:\*\* Box(196,) float32 — agent position, visible disease per cell, inspection mask, treatment mask, resources  

# \*\*Action space:\*\* Discrete(8)  

# \*\*Max steps:\*\* 250  

# 

# \### Actions

# 

# &#x20;   ID  Action         Effect

# &#x20;   0   Move North     Move + auto-inspect new cell

# &#x20;   1   Move South     Move + auto-inspect new cell

# &#x20;   2   Move West      Move + auto-inspect new cell

# &#x20;   3   Move East      Move + auto-inspect new cell

# &#x20;   4   Inspect        Reveal disease level of current cell

# &#x20;   5   Light Treat    1 dose — cures Mild and Moderate

# &#x20;   6   Heavy Treat    2 doses — cures all levels

# &#x20;   7   Mark Healthy   Skip cell, no treatment

# 

# \### Disease Levels

# 

# &#x20;   0 = Healthy (green)

# &#x20;   1 = Mild (yellow)

# &#x20;   2 = Moderate (orange)

# &#x20;   3 = Severe (red)

# &#x20;   4 = Dead (dark grey)

# 

# \### Rewards

# 

# &#x20;   +1.0   new cell inspected

# &#x20;   +2d    disease found (d = severity)

# &#x20;   +10    correct light treatment

# &#x20;   +15    correct heavy treatment

# &#x20;   +5     correct healthy skip

# &#x20;   -0.3   per step (efficiency pressure)

# &#x20;   -3/-5  treating healthy plant (waste)

# &#x20;   -8d    missing diseased cell

# &#x20;   +30 + 20×efficiency   completion bonus

# 

# \---

# 

# \## Quick Start

# 

# &#x20;   # Clone and install

# &#x20;   git clone https://github.com/elyse003/Marie\_Elyse\_Uyiringiye\_rl\_summative.git

# &#x20;   cd marie\_elyse\_rl\_summative

# &#x20;   pip install -r requirements.txt

# 

# &#x20;   # Run best agent

# &#x20;   python main.py --algo ppo --run 0 --episodes 3

# 

# &#x20;   # Random agent demo

# &#x20;   python random\_demo.py --episodes 1

# 

# &#x20;   # Train from scratch

# &#x20;   python -m training.dqn\_training --all --timesteps 300000

# &#x20;   python -m training.pg\_training --algo ppo --all --timesteps 300000

# &#x20;   python -m training.reinforce --all --episodes 2000

# 

# \---

# 

# \## Algorithms

# 

# \*\*DQN\*\* — MLP \[128,128], experience replay, target network, epsilon-greedy  

# Best: Run 9 | lr=5e-4 | gamma=0.99 | buffer=75k | inspected=13.0/64

# 

# \*\*PPO\*\* — Shared \[128,128] MLP, clipped objective, entropy bonus, 4 parallel envs  

# Best: Run 0 | lr=3e-4 | n\_steps=2048 | ent\_coef=0.01 | inspected=38.7/64 ✓

# 

# \*\*REINFORCE\*\* — \[128,128] MLP, Monte Carlo returns, normalisation, entropy bonus  

# Best: Run 0 | lr=1e-3 | gamma=0.99 | normalise=True | entropy=0.01

# 

# \---

# 

# \## Results

# 

# &#x20;   Algorithm   Best Run   Avg Inspected   Avg Reward

# &#x20;   PPO         Run 0      38.7 / 64       -192.3      ← best

# &#x20;   DQN         Run 9      13.0 / 64       -250.6

# &#x20;   REINFORCE   Run 0        —               —

# 

# PPO outperformed DQN because its entropy regularisation explicitly incentivises exploration. DQN converged to a suboptimal exploit — treating the starting cell repeatedly rather than learning full grid coverage.

# 

# \---

# 

# \## Mission Alignment

# 

# Rwanda's smallholder farmers lose 20–40% of yields to undetected crop diseases. This agent autonomously patrols and inspects every plant cell, treating only what needs treating and conserving limited resources — laying the foundation for integration with drone or mobile app hardware.

# 

# \---

# 

# \## Dependencies

# 

# &#x20;   gymnasium>=0.29.0       stable-baselines3>=2.2.0

# &#x20;   torch>=2.0.0            pygame>=2.5.0

# &#x20;   matplotlib>=3.7.0       numpy>=1.24.0

# &#x20;   imageio>=2.31.0         Pillow>=10.0.0

# 

# \---

# 

# \*\*Video Demo:\*\* \[Insert YouTube link]  

# \*\*GitHub:\*\* https://github.com/elyse003/Marie\_Elyse\_Uyiringiye\_rl\_summative.git

