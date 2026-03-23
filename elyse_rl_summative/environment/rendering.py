"""
rendering.py — Pygame-based visualisation for CropDiseaseEnv
=============================================================
Works both interactively (render_mode="human") and headlessly
(render_mode="rgb_array") so it runs on Kaggle / Modal without
a physical display.

Layout
------
┌────────────────────────────────────────────┐
│  HEADER  (title + step counter)            │
├──────────────────┬─────────────────────────┤
│                  │  HUD PANEL              │
│   FARM GRID      │  · Episode stats        │
│   (cells +       │  · Dose bar             │
│    agent)        │  · Step bar             │
│                  │  · Last action          │
│                  │  · Legend               │
├────────────────────────────────────────────┤
│  STATUS BAR  (inspection / treatment pct)  │
└────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import numpy as np
from typing import Optional

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = (18,  20,  30)
HEADER_BG   = (28,  30,  50)
HUD_BG      = (22,  25,  40)
GRID_LINE   = (50,  55,  80)
TEXT_MAIN   = (220, 225, 255)
TEXT_DIM    = (130, 135, 160)
ACCENT      = ( 80, 160, 255)

# Disease level colours  [HEALTHY, MILD, MODERATE, SEVERE, DEAD]
DISEASE_COLOURS = [
    ( 60, 180,  75),   # HEALTHY  — green
    (255, 220,  50),   # MILD     — yellow
    (255, 140,  20),   # MODERATE — orange
    (210,  40,  40),   # SEVERE   — red
    ( 60,  60,  60),   # DEAD     — dark grey
]
UNKNOWN_COLOUR  = ( 40,  45,  65)   # not yet inspected
TREATED_OVERLAY = ( 80, 210, 130)   # green-teal tint for treated cells

AGENT_COLOUR = (255, 255, 255)
AGENT_BORDER = ( 30,  30,  30)

DOSE_BAR_FG  = ( 80, 200, 120)
DOSE_BAR_BG  = ( 50,  55,  70)
STEP_BAR_FG  = (100, 180, 255)
STEP_BAR_BG  = ( 50,  55,  70)

# ── Layout constants ──────────────────────────────────────────────────────────
WINDOW_W    = 900
WINDOW_H    = 640
HEADER_H    = 55
STATUS_H    = 40
HUD_W       = 240
GRID_MARGIN = 18        # px inside the grid area
CELL_PAD    = 2         # px between cells


class CropDiseaseRenderer:
    """Pygame renderer — call render() every step."""

    def __init__(self, grid_size: int = 8, render_mode: str = "human"):
        self.grid_size   = grid_size
        self.render_mode = render_mode
        self._surface    = None
        self._clock      = None
        self._font_lg    = None
        self._font_md    = None
        self._font_sm    = None
        self._last_action_label = "—"
        self._episode_reward    = 0.0
        self._initialised       = False

    # ─────────────────────────────────────────────────────────────────────────
    # Initialise pygame (lazy — only when first render() is called)
    # ─────────────────────────────────────────────────────────────────────────

    def _init_pygame(self):
        if self._initialised:
            return
        # Headless on Kaggle / CI
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        import pygame
        pygame.init()
        if self.render_mode == "human":
            # Try to open a real window; fall back to dummy
            try:
                os.environ.pop("SDL_VIDEODRIVER", None)
                self._surface = pygame.display.set_mode(
                    (WINDOW_W, WINDOW_H), pygame.NOFRAME
                )
                pygame.display.set_caption("CropCare RL — Disease Management")
            except Exception:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                pygame.display.init()
                self._surface = pygame.Surface((WINDOW_W, WINDOW_H))
        else:
            self._surface = pygame.Surface((WINDOW_W, WINDOW_H))

        self._clock  = pygame.time.Clock()
        self._font_lg = pygame.font.SysFont("monospace", 20, bold=True)
        self._font_md = pygame.font.SysFont("monospace", 14, bold=False)
        self._font_sm = pygame.font.SysFont("monospace", 11, bold=False)
        self._initialised = True
        self._pygame = pygame

    # ─────────────────────────────────────────────────────────────────────────
    # Main render entry-point
    # ─────────────────────────────────────────────────────────────────────────

    def render(
        self,
        true_disease: np.ndarray,
        visible_disease: np.ndarray,
        inspected: np.ndarray,
        treated: np.ndarray,
        agent_pos: np.ndarray,
        steps: int,
        max_steps: int,
        doses: int,
        max_doses: int,
        total_reward: float,
        last_action: int = -1,
    ) -> Optional[np.ndarray]:
        self._init_pygame()
        pg = self._pygame

        # Handle quit events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return None

        if last_action >= 0:
            from environment.custom_env import ACTION_LABELS
            self._last_action_label = ACTION_LABELS.get(last_action, str(last_action))

        surf = self._surface
        surf.fill(BG)

        # ── Header ────────────────────────────────────────────────────────
        self._draw_header(surf, steps, max_steps, total_reward)

        # ── Grid area ─────────────────────────────────────────────────────
        grid_area_x = 0
        grid_area_y = HEADER_H
        grid_area_w = WINDOW_W - HUD_W
        grid_area_h = WINDOW_H - HEADER_H - STATUS_H

        self._draw_grid(
            surf,
            true_disease, visible_disease, inspected, treated, agent_pos,
            grid_area_x, grid_area_y, grid_area_w, grid_area_h,
        )

        # ── HUD panel ─────────────────────────────────────────────────────
        hud_x = WINDOW_W - HUD_W
        hud_y = HEADER_H
        self._draw_hud(
            surf, steps, max_steps, doses, max_doses,
            total_reward, inspected, treated, true_disease,
            hud_x, hud_y,
        )

        # ── Status bar ────────────────────────────────────────────────────
        self._draw_status_bar(
            surf, inspected, treated, true_disease, steps, max_steps
        )

        if self.render_mode == "human":
            pg.display.flip()
            self._clock.tick(12)
            return None
        else:
            # Return RGB array for rgb_array mode / video recording
            return np.transpose(
                np.array(pg.surfarray.array3d(surf)), axes=(1, 0, 2)
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-draw helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _draw_header(self, surf, steps, max_steps, total_reward):
        pg = self._pygame
        pg.draw.rect(surf, HEADER_BG, (0, 0, WINDOW_W, HEADER_H))
        pg.draw.line(surf, ACCENT, (0, HEADER_H - 1), (WINDOW_W, HEADER_H - 1), 2)

        title = self._font_lg.render(
            "CropCare RL  •  Crop Disease Management Agent", True, TEXT_MAIN
        )
        surf.blit(title, (14, 10))

        info = self._font_md.render(
            f"Step {steps:>4} / {max_steps}    Reward: {total_reward:>+8.1f}    "
            f"Last action: {self._last_action_label}",
            True, TEXT_DIM,
        )
        surf.blit(info, (14, 33))

    def _draw_grid(
        self, surf,
        true_disease, visible_disease, inspected, treated, agent_pos,
        ax, ay, aw, ah,
    ):
        pg = self._pygame
        G  = self.grid_size
        cell_w = (aw - 2 * GRID_MARGIN) // G
        cell_h = (ah - 2 * GRID_MARGIN) // G
        ox = ax + GRID_MARGIN
        oy = ay + GRID_MARGIN

        for r in range(G):
            for c in range(G):
                x = ox + c * cell_w
                y = oy + r * cell_h
                rect = pg.Rect(x + CELL_PAD, y + CELL_PAD,
                               cell_w - CELL_PAD * 2, cell_h - CELL_PAD * 2)

                vis = visible_disease[r, c]
                if vis < 0:
                    base_col = UNKNOWN_COLOUR
                else:
                    base_col = DISEASE_COLOURS[vis]

                # Treated cells get a slight overlay tint
                if treated[r, c] == 1:
                    base_col = _blend(base_col, TREATED_OVERLAY, 0.35)

                pg.draw.rect(surf, base_col, rect, border_radius=4)

                # Grid lines
                pg.draw.rect(surf, GRID_LINE, rect, 1, border_radius=4)

                # Inspection indicator — small dot top-right
                if inspected[r, c] > 0:
                    dot_x = x + cell_w - CELL_PAD - 6
                    dot_y = y + CELL_PAD + 6
                    pg.draw.circle(surf, TEXT_MAIN, (dot_x, dot_y), 3)

                # Treatment cross symbol
                if treated[r, c] == 1:
                    cx_, cy_ = x + cell_w // 2, y + cell_h // 2
                    half = cell_w // 5
                    pg.draw.line(surf, (255,255,255), (cx_-half, cy_), (cx_+half, cy_), 2)
                    pg.draw.line(surf, (255,255,255), (cx_, cy_-half), (cx_, cy_+half), 2)

                # Disease severity label
                if vis > 0:
                    lbl = self._font_sm.render(str(vis), True, (255, 255, 255))
                    surf.blit(lbl, (x + 4, y + 4))

        # ── Draw agent ────────────────────────────────────────────────────
        ar, ac = agent_pos
        ax_ = ox + ac * cell_w + cell_w // 2
        ay_ = oy + ar * cell_h + cell_h // 2
        radius = min(cell_w, cell_h) // 3
        pg.draw.circle(surf, AGENT_BORDER, (ax_, ay_), radius + 2)
        pg.draw.circle(surf, AGENT_COLOUR, (ax_, ay_), radius)
        # Arrow pointing down (like an agent "facing" direction)
        pg.draw.polygon(
            surf, ACCENT,
            [(ax_, ay_ + radius // 2 + 4),
             (ax_ - radius // 3, ay_),
             (ax_ + radius // 3, ay_)],
        )

    def _draw_hud(
        self, surf, steps, max_steps, doses, max_doses,
        total_reward, inspected, treated, true_disease,
        hx, hy,
    ):
        pg = self._pygame
        hh = WINDOW_H - HEADER_H - STATUS_H
        pg.draw.rect(surf, HUD_BG, (hx, hy, HUD_W, hh))
        pg.draw.line(surf, GRID_LINE, (hx, hy), (hx, hy + hh), 1)

        G = self.grid_size
        total_cells    = G * G
        n_inspected    = int(np.sum(inspected > 0))
        n_treated      = int(np.sum(treated == 1))
        n_sick         = int(np.sum(true_disease > 0))

        y = hy + 14
        pad = 12

        def label(text, colour=TEXT_DIM, font=None):
            nonlocal y
            f = font or self._font_sm
            t = f.render(text, True, colour)
            surf.blit(t, (hx + pad, y))
            y += t.get_height() + 4

        def section(title):
            nonlocal y
            y += 6
            t = self._font_md.render(title, True, ACCENT)
            surf.blit(t, (hx + pad, y))
            y += t.get_height() + 2
            pg.draw.line(surf, GRID_LINE,
                         (hx + pad, y), (hx + HUD_W - pad, y), 1)
            y += 6

        def progress_bar(val, mx, fg, bg, label_str):
            nonlocal y
            bar_x  = hx + pad
            bar_w  = HUD_W - pad * 2
            bar_h  = 14
            pct    = val / max(mx, 1)
            pg.draw.rect(surf, bg,  (bar_x, y, bar_w, bar_h), border_radius=3)
            pg.draw.rect(surf, fg,  (bar_x, y, int(bar_w * pct), bar_h), border_radius=3)
            lbl = self._font_sm.render(label_str, True, TEXT_MAIN)
            surf.blit(lbl, (bar_x + 4, y + 1))
            y += bar_h + 6

        section("Episode Stats")
        label(f"Total Reward : {total_reward:>+.1f}", TEXT_MAIN)
        label(f"Sick cells   : {n_sick:>3} / {total_cells}")
        label(f"Inspected    : {n_inspected:>3} / {total_cells}")
        label(f"Treated      : {n_treated:>3}")

        section("Resources")
        progress_bar(doses, max_doses, DOSE_BAR_FG, DOSE_BAR_BG,
                     f"Doses {doses}/{max_doses}")
        progress_bar(max_steps - steps, max_steps, STEP_BAR_FG, STEP_BAR_BG,
                     f"Time  {max_steps-steps}/{max_steps}")

        section("Legend")
        for level, name in enumerate(["Healthy","Mild","Moderate","Severe","Dead"]):
            col  = DISEASE_COLOURS[level]
            rect = pg.Rect(hx + pad, y + 1, 12, 12)
            pg.draw.rect(surf, col, rect, border_radius=2)
            label(f"  {level} {name}")
        label("  ⬜ Unknown / uninspected")
        label("  ✚ Treated cell", TEXT_MAIN)
        label("  ● Agent position", TEXT_MAIN)

    def _draw_status_bar(
        self, surf, inspected, treated, true_disease, steps, max_steps
    ):
        pg = self._pygame
        G  = self.grid_size
        total  = G * G
        n_insp = int(np.sum(inspected > 0))
        n_trt  = int(np.sum(treated == 1))
        n_sick = int(np.sum(true_disease > 0))

        sy = WINDOW_H - STATUS_H
        pg.draw.rect(surf, HEADER_BG, (0, sy, WINDOW_W, STATUS_H))
        pg.draw.line(surf, ACCENT, (0, sy), (WINDOW_W, sy), 1)

        pct_insp = 100 * n_insp // total
        pct_trt  = 100 * n_trt  // max(n_sick, 1)

        text = (
            f"Inspection {pct_insp:>3}%  ({n_insp}/{total})   │   "
            f"Treatment coverage {pct_trt:>3}%  ({n_trt}/{n_sick} diseased)   │   "
            f"Step {steps}/{max_steps}"
        )
        t = self._font_md.render(text, True, TEXT_DIM)
        surf.blit(t, (14, sy + (STATUS_H - t.get_height()) // 2))

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def close(self):
        if self._initialised:
            self._pygame.quit()
            self._initialised = False


# ─── Utilities ────────────────────────────────────────────────────────────────

def _blend(
    c1: tuple, c2: tuple, alpha: float
) -> tuple:
    """Linear blend: (1-alpha)*c1 + alpha*c2"""
    return tuple(
        int((1 - alpha) * a + alpha * b)
        for a, b in zip(c1, c2)
    )
