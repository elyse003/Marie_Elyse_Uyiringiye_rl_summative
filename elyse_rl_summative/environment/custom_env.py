"""
CropCare RL Environment
=======================
A farm simulation where an agent navigates a grid of crops,
inspects plants for disease, and applies treatments to maximize
farm health while minimising resource usage.

Mission: Detecting and managing crop diseases for smallholder
farmers — aligned with the CropCare capstone project.

Grid cells have a hidden true disease state. The agent must
INSPECT cells to reveal their state, then choose the correct
treatment action. Disease spreads stochastically, rewarding
efficient, prioritised inspection/treatment.

Action Space  (Discrete 8):
  0 - Move North      4 - Inspect cell
  1 - Move South      5 - Light Treatment  (1 dose, cures Mild/Moderate)
  2 - Move West       6 - Heavy Treatment  (2 doses, cures all levels)
  3 - Move East       7 - Mark as Healthy  (skip, no treatment)

Observation Space (Box, float32):
  [agent_row, agent_col,                     (2)
   disease_obs  (grid_size^2, normalised),   (64)
   inspect_mask (grid_size^2),               (64)
   treated_mask (grid_size^2),               (64)
   doses_remaining, steps_remaining]         (2)
  Total: 196 features for 8×8 grid

Reward:
  +1.0  each new inspection
  +2*d  bonus per disease severity d found
  +10   light-treat mild/moderate correctly
  +15   heavy-treat severe/dead correctly
  +5    correctly mark healthy cell
  -3    light-treat healthy (waste)
  -5    heavy-treat healthy (waste)
  -8*d  miss diseased cell (mark_healthy error)
  -0.5  per step (efficiency pressure)
  -1    wall collision or redundant action
  +30+20*efficiency  completion bonus

Terminal Conditions:
  Terminated : all cells inspected
  Truncated  : max_steps reached
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ─── Disease level constants ────────────────────────────────────────────────
HEALTHY  = 0
MILD     = 1
MODERATE = 2
SEVERE   = 3
DEAD     = 4

# ─── Action constants ────────────────────────────────────────────────────────
ACT_NORTH        = 0
ACT_SOUTH        = 1
ACT_WEST         = 2
ACT_EAST         = 3
ACT_INSPECT      = 4
ACT_LIGHT_TREAT  = 5
ACT_HEAVY_TREAT  = 6
ACT_MARK_HEALTHY = 7

ACTION_LABELS = {
    ACT_NORTH:        "Move North",
    ACT_SOUTH:        "Move South",
    ACT_WEST:         "Move West",
    ACT_EAST:         "Move East",
    ACT_INSPECT:      "Inspect",
    ACT_LIGHT_TREAT:  "Light Treat",
    ACT_HEAVY_TREAT:  "Heavy Treat",
    ACT_MARK_HEALTHY: "Mark Healthy",
}


class CropDiseaseEnv(gym.Env):
    """Custom Gymnasium environment for crop disease management."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

    def __init__(
        self,
        grid_size: int = 8,
        max_steps: int = 250,
        initial_disease_prob: float = 0.35,
        max_treatment_doses: int = 30,
        spread_probability: float = 0.04,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.grid_size          = grid_size
        self.max_steps          = max_steps
        self.initial_disease_prob = initial_disease_prob
        self.max_treatment_doses  = max_treatment_doses
        self.spread_probability   = spread_probability
        self.render_mode          = render_mode

        # ── Action space ──────────────────────────────────────────────────
        self.action_space = spaces.Discrete(8)

        # ── Observation space ─────────────────────────────────────────────
        #   2 + 3*(grid^2) + 2 = 196 for 8×8
        n = grid_size * grid_size
        obs_size = 2 + n * 3 + 2
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # internal state (set in reset)
        self._true_disease: np.ndarray = None
        self._visible_disease: np.ndarray = None
        self._inspected: np.ndarray = None
        self._treated: np.ndarray = None
        self._agent_pos: np.ndarray = None
        self._steps: int = 0
        self._doses: int = 0
        self._total_reward: float = 0.0
        self._episode_correct_treats: int = 0
        self._episode_missed: int = 0

        self._renderer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        rng = self.np_random

        G = self.grid_size

        # True disease levels — hidden from the agent until inspected
        self._true_disease = np.zeros((G, G), dtype=np.int32)
        mask = rng.random((G, G)) < self.initial_disease_prob
        levels = rng.integers(1, 5, size=(G, G))        # 1-4
        self._true_disease[mask] = levels[mask]

        # Agent observation layers
        self._visible_disease = np.full((G, G), -1, dtype=np.int32)   # -1 = unknown
        self._inspected = np.zeros((G, G), dtype=np.int32)
        self._treated   = np.zeros((G, G), dtype=np.int32)

        # Agent starts at (0, 0) — top-left corner
        self._agent_pos = np.array([0, 0], dtype=np.int32)

        self._steps  = 0
        self._doses  = self.max_treatment_doses
        self._total_reward = 0.0
        self._episode_correct_treats = 0
        self._episode_missed = 0

        # Auto-inspect the starting cell
        self._do_inspect(0, 0)

        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        r, c = self._agent_pos
        reward = -0.5          # per-step cost (encourage efficiency)
        terminated = False
        truncated  = False

        # ── Movement ──────────────────────────────────────────────────────
        if action in (ACT_NORTH, ACT_SOUTH, ACT_WEST, ACT_EAST):
            dr, dc = {
                ACT_NORTH: (-1,  0),
                ACT_SOUTH: ( 1,  0),
                ACT_WEST:  ( 0, -1),
                ACT_EAST:  ( 0,  1),
            }[action]
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                self._agent_pos[:] = [nr, nc]
                reward += 0.1          # valid move bonus
            else:
                reward -= 1.0          # wall collision penalty

        # ── Inspect ───────────────────────────────────────────────────────
        elif action == ACT_INSPECT:
            if self._inspected[r, c] == 0:
                self._do_inspect(r, c)
                reward += 1.0
                d = self._true_disease[r, c]
                if d > 0:
                    reward += 2.0 * d  # bigger bonus for finding worse disease
            else:
                reward -= 0.5          # re-inspection penalty

        # ── Light Treatment ───────────────────────────────────────────────
        elif action == ACT_LIGHT_TREAT:
            reward += self._apply_treatment(r, c, heavy=False)

        # ── Heavy Treatment ───────────────────────────────────────────────
        elif action == ACT_HEAVY_TREAT:
            reward += self._apply_treatment(r, c, heavy=True)

        # ── Mark as Healthy ───────────────────────────────────────────────
        elif action == ACT_MARK_HEALTHY:
            if self._inspected[r, c] == 0:
                reward -= 1.0          # must inspect before deciding
            else:
                d = self._true_disease[r, c]
                if d == HEALTHY or self._treated[r, c] == 1:
                    reward += 5.0
                else:
                    reward -= 8.0 * d  # severe penalty for missing disease
                    self._episode_missed += 1
                self._inspected[r, c] = 2   # mark "decided"

        # ── Disease spreads stochastically ────────────────────────────────
        self._spread_disease()

        self._steps += 1
        self._total_reward += reward

        # ── Termination check ─────────────────────────────────────────────
        all_decided = bool(np.all(self._inspected > 0))
        if all_decided:
            terminated = True
            efficiency  = self._doses / self.max_treatment_doses
            reward += 30.0 + 20.0 * efficiency

        if self._steps >= self.max_steps:
            truncated = True
            undecided_sick = int(np.sum(
                (self._true_disease > 0) & (self._inspected == 0)
            ))
            reward -= 2.0 * undecided_sick

        obs  = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        if self._renderer is None:
            from environment.rendering import CropDiseaseRenderer
            self._renderer = CropDiseaseRenderer(
                self.grid_size, render_mode=self.render_mode
            )
        return self._renderer.render(
            true_disease=self._true_disease,
            visible_disease=self._visible_disease,
            inspected=self._inspected,
            treated=self._treated,
            agent_pos=self._agent_pos,
            steps=self._steps,
            max_steps=self.max_steps,
            doses=self._doses,
            max_doses=self.max_treatment_doses,
            total_reward=self._total_reward,
        )

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _do_inspect(self, r: int, c: int):
        self._inspected[r, c] = 1
        self._visible_disease[r, c] = self._true_disease[r, c]

    def _apply_treatment(self, r: int, c: int, heavy: bool) -> float:
        doses_needed = 2 if heavy else 1
        if self._doses < doses_needed:
            return -2.0                    # out of doses
        if self._inspected[r, c] == 0:
            return -1.0                    # treat without inspecting
        if self._treated[r, c] == 1:
            return -1.0                    # already treated

        d = self._true_disease[r, c]
        self._doses -= doses_needed
        self._treated[r, c] = 1

        if d == HEALTHY:
            return -5.0 if heavy else -3.0   # wasted on healthy plant

        if heavy:
            # Heavy treatment cures all disease levels
            self._true_disease[r, c] = HEALTHY
            self._visible_disease[r, c] = HEALTHY
            self._episode_correct_treats += 1
            return 8.0 if d <= MODERATE else 15.0

        else:
            # Light treatment cures mild/moderate; partial on severe/dead
            if d <= MODERATE:
                self._true_disease[r, c] = HEALTHY
                self._visible_disease[r, c] = HEALTHY
                self._episode_correct_treats += 1
                return 10.0
            else:
                # Reduces severity by 1
                self._true_disease[r, c] = d - 1
                self._visible_disease[r, c] = d - 1
                return 3.0

    def _spread_disease(self):
        """Stochastic disease spread and progression."""
        rng   = self.np_random
        G     = self.grid_size
        new_d = self._true_disease.copy()

        for r in range(G):
            for c in range(G):
                level = self._true_disease[r, c]
                if level == 0 or self._treated[r, c] == 1:
                    continue
                # Spread to untreated neighbours
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < G and 0 <= nc < G:
                        if new_d[nr, nc] == 0 and self._treated[nr, nc] == 0:
                            if rng.random() < self.spread_probability:
                                new_d[nr, nc] = MILD
                                if self._inspected[nr, nc]:
                                    self._visible_disease[nr, nc] = MILD
                # Progression: untreated severe/dead cells can worsen
                if level >= SEVERE and rng.random() < 0.02:
                    new_d[r, c] = min(DEAD, level + 1)
                    if self._inspected[r, c]:
                        self._visible_disease[r, c] = new_d[r, c]

        self._true_disease = new_d

    def _get_obs(self) -> np.ndarray:
        G = self.grid_size
        r, c = self._agent_pos

        pos = np.array(
            [r / (G - 1), c / (G - 1)], dtype=np.float32
        )

        disease_obs = np.where(
            self._visible_disease >= 0,
            self._visible_disease / 4.0,
            0.0,
        ).flatten().astype(np.float32)

        inspect_obs  = (self._inspected > 0).flatten().astype(np.float32)
        treated_obs  = self._treated.flatten().astype(np.float32)

        resources = np.array(
            [
                self._doses / self.max_treatment_doses,
                (self.max_steps - self._steps) / self.max_steps,
            ],
            dtype=np.float32,
        )

        return np.concatenate([pos, disease_obs, inspect_obs, treated_obs, resources])

    def _get_info(self) -> Dict[str, Any]:
        total_diseased   = int(np.sum(self._true_disease > 0))
        total_inspected  = int(np.sum(self._inspected > 0))
        total_treated    = int(np.sum(self._treated == 1))
        healthy_original = int(np.sum(self._true_disease == 0))

        return {
            "steps":             self._steps,
            "doses_remaining":   self._doses,
            "total_cells":       self.grid_size ** 2,
            "total_diseased":    total_diseased,
            "cells_inspected":   total_inspected,
            "cells_treated":     total_treated,
            "correct_treats":    self._episode_correct_treats,
            "missed_diseases":   self._episode_missed,
            "total_reward":      round(self._total_reward, 2),
            "agent_pos":         self._agent_pos.tolist(),
            "completion_pct":    round(100 * total_inspected / self.grid_size**2, 1),
        }

    # ─── Pretty print for debugging ──────────────────────────────────────────

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CropDiseaseEnv(grid={self.grid_size}x{self.grid_size}, "
            f"max_steps={self.max_steps}, "
            f"disease_prob={self.initial_disease_prob})"
        )
