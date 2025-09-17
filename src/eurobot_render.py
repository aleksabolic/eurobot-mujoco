# eurobot_render.py
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional
from eurobot_world import EurobotWorld, NodeType, Col

# Choose a fixed 3m x 2m table extent centered at (0,0) – matches your PNG aspect
TABLE_X_MIN, TABLE_X_MAX = -1.5, 1.5
TABLE_Y_MIN, TABLE_Y_MAX = -1.0, 1.0

# Patch sizes (meters)
PANTRY_HALF = 0.10        # 20x20 cm zones
PICKUP_R    = 0.075       # ~7.5 cm marker radius
NEST_HALF_X = 0.30        # 60x45 cm nest (half-sizes)
NEST_HALF_Y = 0.225

BLUE_COLOR   = (0.20, 0.40, 1.00)
YELLOW_COLOR = (1.00, 0.90, 0.20)

class EurobotRenderer:
    """
    Snapshot renderer: call draw_snapshot(world, idx=-1) to render a single state.
    If idx == -1, draws the current (latest) snapshot in world.history,
    and if history is empty, it draws from live world state.
    """
    def __init__(self,
                 world: EurobotWorld,
                 bg_path: str = "assets/table_bis.png",
                 bg_alpha: float = 0.35,
                 figsize=(12, 8)):
        self.world = world
        self.bg_path = bg_path
        self.bg_alpha = float(bg_alpha)

        # prepare figure/axes once
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_xlim(TABLE_X_MIN, TABLE_X_MAX)
        self.ax.set_ylim(TABLE_Y_MIN, TABLE_Y_MAX)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(np.linspace(TABLE_X_MIN, TABLE_X_MAX, 7))
        self.ax.set_yticks(np.linspace(TABLE_Y_MIN, TABLE_Y_MAX, 5))
        self.ax.grid(True, linewidth=0.3, alpha=0.3)

        # background image (if present)
        self._bg_im = None
        if os.path.exists(self.bg_path):
            img = plt.imread(self.bg_path)
            self._bg_im = self.ax.imshow(
                img, extent=(TABLE_X_MIN, TABLE_X_MAX, TABLE_Y_MIN, TABLE_Y_MAX),
                origin="upper", alpha=self.bg_alpha, zorder=-5
            )

        # optional: draw a thin table boundary
        self.ax.add_patch(patches.Rectangle(
            (TABLE_X_MIN, TABLE_Y_MIN),
            TABLE_X_MAX - TABLE_X_MIN, TABLE_Y_MAX - TABLE_Y_MIN,
            linewidth=1.0, edgecolor='black', facecolor='none', zorder=-4
        ))

        # cache of dynamic artists so we can clear them each draw
        self._dynamic_artists = []

        # Title/status text
        self._title = self.ax.text(0.01, 1.02, "", transform=self.ax.transAxes, fontsize=11)

    # ----------------- public API -----------------
    def draw_snapshot(self, idx: int = -1, show: bool = False, pause: Optional[float] = None):
        """
        Draw a single snapshot (state) from world.history.
        - idx: which snapshot; -1 means last (most recent). If no history, uses live world.
        - show: call plt.show() at the end (blocking). Good for one-off renders.
        - pause: if not None, calls plt.pause(pause) to update non-blocking.
        """
        # pick data source
        if self.world.history:
            snap = self.world.history[idx]
            blue_node   = snap["blue_node"]
            yellow_node = snap["yellow_node"]
            t_left      = snap["t_left"]
            pantries    = snap["pantries"]
            pickups     = snap["pickups"]
            blue_inv    = snap["blue_inv"]
            yellow_inv  = snap["yellow_inv"]
        else:
            # live fallback (if history not recorded yet)
            blue_node   = self.world.blue.node
            yellow_node = self.world.yellow.node
            t_left      = self.world.t_left
            pantries    = self.world.pantries
            pickups     = self.world.pickups
            blue_inv    = self.world.blue.inv
            yellow_inv  = self.world.yellow.inv

        # clear old dynamic artists
        for art in self._dynamic_artists:
            try: art.remove()
            except Exception: pass
        self._dynamic_artists.clear()

        # Title
        self._title.set_text(f"t_left = {t_left:.1f}s")

        # --- draw nests (rectangles around named nodes) ---
        for i, node in enumerate(self.world.nodes):
            if node.kind == NodeType.NEST:
                rect = patches.Rectangle(
                    (node.xy[0] - NEST_HALF_X, node.xy[1] - NEST_HALF_Y),
                    2*NEST_HALF_X, 2*NEST_HALF_Y,
                    linewidth=1.2,
                    edgecolor='black',
                    facecolor='none',
                    zorder=-1
                )
                self.ax.add_patch(rect)
                self._dynamic_artists.append(rect)

        # --- draw pantries ---
        for k, idx_node in enumerate(self.world.PANTRIES):
            node = self.world.nodes[idx_node]
            b, y, n = int(pantries[k, Col.BLUE]), int(pantries[k, Col.YELLOW]), int(pantries[k, Col.NEUTRAL])
            # majority color for fill
            if b > y:
                face = (0.75, 0.82, 1.0)   # light blue
            elif y > b:
                face = (1.0, 0.98, 0.70)   # light yellow
            else:
                face = (0.85, 0.95, 0.95)  # neutral (cyan-ish)

            rect = patches.Rectangle(
                (node.xy[0] - PANTRY_HALF, node.xy[1] - PANTRY_HALF),
                2*PANTRY_HALF, 2*PANTRY_HALF,
                linewidth=1.0, edgecolor='black', facecolor=face, zorder=0
            )
            self.ax.add_patch(rect)
            self._dynamic_artists.append(rect)

            txt = self.ax.text(
                node.xy[0], node.xy[1],
                f"B{b}/Y{y}/N{n}",
                ha="center", va="center", fontsize=8
            )
            self._dynamic_artists.append(txt)

        # --- draw pickups ---
        for k, idx_node in enumerate(self.world.PICKUPS):
            node = self.world.nodes[idx_node]
            b, y = int(pickups[k, Col.BLUE]), int(pickups[k, Col.YELLOW])
            available = (b + y) > 0
            face = (0.95, 0.75, 0.95) if available else (0.85, 0.85, 0.85)
            circ = patches.Circle(
                (node.xy[0], node.xy[1]), PICKUP_R,
                linewidth=1.0, edgecolor='black', facecolor=face, zorder=0
            )
            self.ax.add_patch(circ)
            self._dynamic_artists.append(circ)

            txt = self.ax.text(
                node.xy[0], node.xy[1] - 0.04,
                f"B{b} Y{y}", ha="center", va="center", fontsize=8
            )
            self._dynamic_artists.append(txt)

        # --- draw robots ---
        bx, by = self.world.nodes[blue_node].xy
        yx, yy = self.world.nodes[yellow_node].xy

        blue_dot = self.ax.plot(bx, by, 'o', markersize=25, color=BLUE_COLOR, zorder=1)[0]
        yellow_dot = self.ax.plot(yx, yy, 'o', markersize=25, color=YELLOW_COLOR, zorder=1)[0]
        self._dynamic_artists.extend([blue_dot, yellow_dot])

        # inventories text (top-left corner)
        bl_text = self.ax.text(
            TABLE_X_MIN + 0.05, TABLE_Y_MAX - 0.05,
            f"BLUE inv: B{int(blue_inv[Col.BLUE])} Y{int(blue_inv[Col.YELLOW])} N{int(blue_inv[Col.NEUTRAL])}",
            fontsize=9, ha='left', va='top'
        )
        yl_text = self.ax.text(
            TABLE_X_MIN + 0.05, TABLE_Y_MAX - 0.13,
            f"YELL inv: B{int(yellow_inv[Col.BLUE])} Y{int(yellow_inv[Col.YELLOW])} N{int(yellow_inv[Col.NEUTRAL])}",
            fontsize=9, ha='left', va='top'
        )
        self._dynamic_artists.extend([bl_text, yl_text])

        # draw now
        if pause is not None:
            plt.pause(pause)
        elif show:
            plt.show()
        else:
            self.fig.canvas.draw_idle()
