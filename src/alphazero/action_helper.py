from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from eurobot_world import NEST_CAP
from robot import Verb


@dataclass
class ObservationContext:
    blue_node: int
    inv_b: np.ndarray  # shape (n_colors,)
    inv_sum: float
    cap_left: float
    pan: np.ndarray  # shape (n_pantries, n_colors)
    pick: np.ndarray  # shape (n_pickups, n_colors)
    pan_room: np.ndarray  # shape (n_pantries,)
    pan_sum: np.ndarray  # shape (n_pantries,)
    pick_sum: np.ndarray  # shape (n_pickups,)
    nest_blue: float
    nest_room: float


class EurobotActionHelper:
    """Utility to enumerate, encode, and validate actions for AlphaZero."""

    def __init__(self, env) -> None:
        space = env.action_space
        nvec = getattr(space, "nvec", None)
        if nvec is None:
            raise ValueError("EurobotActionHelper requires a MultiDiscrete action space.")
        world = env.world
        self.n_pantries = len(world.PANTRIES)
        self.n_pickups = len(world.PICKUPS)
        self.capacity = float(world.blue.profile.capacity)
        self.can_flip = bool(world.blue.profile.can_flip)
        self.allow_steal = bool(world.allow_steal)
        self.pantry_cap = float(world.pantry_cap)
        self.nest_blue_node = int(world.NEST_BLUE)
        self.pantry_idx = np.asarray(world.pantry_idx, dtype=np.int64)
        self.pickup_idx = np.asarray(world.pickup_idx, dtype=np.int64)
        self.pantry_nodes: Sequence[int] = list(world.PANTRIES)
        self.pickup_nodes: Sequence[int] = list(world.PICKUPS)
        self.max_qty = int(world.blue.profile.max_action_qty)

        self.n_verbs = int(nvec[0])
        self.n_nodes = int(nvec[1])
        self.n_colors = int(nvec[2])

        # Observation slices mirror EurobotDiscreteEnv layout.
        obs_dim = env.observation_space.shape[0]
        i = 0
        self.sl_node_blue = slice(i, i + self.n_nodes)
        i += self.n_nodes
        self.sl_node_yellow = slice(i, i + self.n_nodes)
        i += self.n_nodes
        self.sl_inv_b = slice(i, i + self.n_colors)
        i += self.n_colors
        self.sl_pan = slice(i, i + self.n_colors * self.n_pantries)
        i += self.n_colors * self.n_pantries
        self.sl_pick = slice(i, i + self.n_colors * self.n_pickups)
        i += self.n_colors * self.n_pickups
        self.sl_nest = slice(i, i + 2)
        assert i <= obs_dim

        # Pre-compute action lookup tables.
        self.actions: List[Tuple[int, int, int, int]] = []
        for verb in range(self.n_verbs):
            for node in range(self.n_nodes):
                for color in range(self.n_colors):
                    for qty in range(self.max_qty + 1):
                        self.actions.append((verb, node, color, qty))
        self.action_to_index: Dict[Tuple[int, int, int, int], int] = {
            action: idx for idx, action in enumerate(self.actions)
        }

    # ------------------------------------------------------------------ #
    # Encoding helpers
    def encode(self, action: Tuple[int, int, int, int]) -> int:
        return int(self.action_to_index[action])

    def decode(self, index: int) -> Tuple[int, int, int, int]:
        return self.actions[int(index)]

    def size(self) -> int:
        return len(self.actions)

    # ------------------------------------------------------------------ #
    def _ctx(self, obs: np.ndarray) -> ObservationContext:
        obs = np.asarray(obs, dtype=np.float32)
        blue_slice = obs[self.sl_node_blue]
        blue_node = int(np.argmax(blue_slice))

        inv_b = obs[self.sl_inv_b].copy()
        inv_sum = float(inv_b.sum())
        cap_left = max(0.0, self.capacity - inv_sum)

        if self.n_pantries > 0:
            pan_flat = obs[self.sl_pan]
            pan = pan_flat.reshape(self.n_pantries, self.n_colors).copy()
            pan_sum = pan.sum(axis=1)
            pan_room = np.clip(self.pantry_cap - pan_sum, a_min=0.0, a_max=None)
        else:
            pan = np.zeros((0, self.n_colors), dtype=np.float32)
            pan_sum = np.zeros((0,), dtype=np.float32)
            pan_room = np.zeros((0,), dtype=np.float32)

        if self.n_pickups > 0:
            pick_flat = obs[self.sl_pick]
            pick = pick_flat.reshape(self.n_pickups, self.n_colors).copy()
            pick_sum = pick.sum(axis=1)
        else:
            pick = np.zeros((0, self.n_colors), dtype=np.float32)
            pick_sum = np.zeros((0,), dtype=np.float32)

        if self.sl_nest.stop > self.sl_nest.start:
            nest_counts = obs[self.sl_nest]
            nest_blue = float(nest_counts[0])
        else:
            nest_blue = 0.0

        nest_room = max(0.0, float(NEST_CAP) - nest_blue)

        return ObservationContext(
            blue_node=blue_node,
            inv_b=inv_b,
            inv_sum=inv_sum,
            cap_left=cap_left,
            pan=pan,
            pick=pick,
            pan_room=pan_room,
            pan_sum=pan_sum,
            pick_sum=pick_sum,
            nest_blue=nest_blue,
            nest_room=nest_room,
        )

    # ------------------------------------------------------------------ #
    def _valid_pick(self, ctx: ObservationContext, node: int, color: int, qty: int) -> bool:
        if qty <= 0 or self.n_pickups == 0:
            return False
        if ctx.cap_left <= 0.0:
            return False
        local = int(self.pickup_idx[node]) if node < len(self.pickup_idx) else -1
        if local < 0 or local >= self.n_pickups:
            return False
        stock_total = float(ctx.pick_sum[local])
        if stock_total <= 0.0:
            return False
        stock_color = float(ctx.pick[local, color])
        if stock_color <= 0.0:
            return False
        limit = min(stock_color, ctx.cap_left, float(self.max_qty))
        return 1 <= qty <= int(limit + 1e-6)

    def _valid_place(self, ctx: ObservationContext, node: int, color: int, qty: int) -> bool:
        if qty <= 0 or ctx.inv_b[color] <= 0.0:
            return False
        qty = int(qty)

        if node == self.nest_blue_node:
            if ctx.nest_room <= 0.0:
                return False
            limit = min(ctx.nest_room, ctx.inv_b[color], float(self.max_qty))
            return 1 <= qty <= int(limit + 1e-6)

        local = int(self.pantry_idx[node]) if node < len(self.pantry_idx) else -1
        if local < 0 or local >= self.n_pantries:
            return False
        room = float(ctx.pan_room[local])
        if room <= 0.0:
            return False
        limit = min(room, ctx.inv_b[color], float(self.max_qty))
        return 1 <= qty <= int(limit + 1e-6)

    def _valid_flip(self, ctx: ObservationContext, node: int, color: int, qty: int) -> bool:
        if not self.can_flip or qty <= 0:
            return False
        if node != ctx.blue_node:
            return False
        other_color = 1 - int(color)
        stock_other = float(ctx.inv_b[other_color])
        if stock_other <= 0.0:
            return False
        limit = min(stock_other, float(self.max_qty))
        return 1 <= qty <= int(limit + 1e-6)

    def _valid_steal(self, ctx: ObservationContext, node: int, color: int, qty: int) -> bool:
        if not self.allow_steal or qty <= 0 or ctx.cap_left <= 0.0:
            return False
        local = int(self.pantry_idx[node]) if node < len(self.pantry_idx) else -1
        if local < 0 or local >= self.n_pantries:
            return False
        stock = float(ctx.pan[local, color])
        if stock <= 0.0:
            return False
        limit = min(stock, ctx.cap_left, float(self.max_qty))
        return 1 <= qty <= int(limit + 1e-6)

    def _valid_verb(self, ctx: ObservationContext, verb: int) -> bool:
        verb_enum = Verb(int(verb))
        if verb_enum == Verb.PICK:
            if self.n_pickups == 0 or ctx.cap_left <= 0.0:
                return False
            return bool(np.any(ctx.pick_sum > 0.0))
        if verb_enum == Verb.PLACE:
            if ctx.inv_sum <= 0.0:
                return False
            pantry_room_any = bool(np.any(ctx.pan_room > 0.0)) if self.n_pantries > 0 else False
            nest_room = ctx.nest_room > 0.0
            return pantry_room_any or nest_room
        if verb_enum == Verb.FLIP:
            if not self.can_flip:
                return False
            other_color_stock = ctx.inv_sum - ctx.inv_b
            return bool(np.any(other_color_stock > 0.0))
        if verb_enum == Verb.STEAL:
            if not self.allow_steal or self.n_pantries == 0 or ctx.cap_left <= 0.0:
                return False
            if ctx.pan.shape[0] > 0:
                steal_col = 1 if ctx.pan.shape[1] > 1 else 0
                opp_stock = ctx.pan[:, steal_col]
            else:
                opp_stock = np.zeros((0,), dtype=np.float32)
            return bool(np.any(opp_stock > 0.0))
        return False

    # ------------------------------------------------------------------ #
    def legal_action_mask(self, obs: np.ndarray) -> np.ndarray:
        ctx = self._ctx(obs)
        mask = np.zeros(len(self.actions), dtype=bool)

        valid_verbs = {verb for verb in range(self.n_verbs) if self._valid_verb(ctx, verb)}

        for idx, (verb, node, color, qty) in enumerate(self.actions):
            if verb not in valid_verbs:
                continue

            verb_enum = Verb(int(verb))
            if verb_enum == Verb.PICK:
                if self._valid_pick(ctx, node, color, qty):
                    mask[idx] = True
            elif verb_enum == Verb.PLACE:
                if self._valid_place(ctx, node, color, qty):
                    mask[idx] = True
            elif verb_enum == Verb.FLIP:
                if self._valid_flip(ctx, node, color, qty):
                    mask[idx] = True
            elif verb_enum == Verb.STEAL:
                if self._valid_steal(ctx, node, color, qty):
                    mask[idx] = True

        return mask

    def legal_actions(self, obs: np.ndarray) -> List[int]:
        mask = self.legal_action_mask(obs)
        return [idx for idx, is_valid in enumerate(mask) if is_valid]

    def filter_illegal(self, probs: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Apply legal action mask to probability vector and renormalize."""
        probs = np.asarray(probs, dtype=np.float32)
        mask = self.legal_action_mask(obs)
        clipped = np.where(mask, probs, 0.0)
        total = float(clipped.sum())
        if total <= 0.0:
            return mask.astype(np.float32) / max(mask.sum(), 1)
        return clipped / total
