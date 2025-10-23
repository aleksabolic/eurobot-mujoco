from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

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
        self.actions: list[tuple[int,int,int,int]] = []
        self.per_verb_indices: list[list[int]] = []  # flat indices per verb

        flat_idx = 0
        for v in range(self.n_verbs):
            verb = Verb(v)
            if verb == Verb.PICK:
                nodes = self.pickup_nodes
            elif verb == Verb.PLACE:
                nodes = self.pantry_nodes + [self.nest_blue_node]
            elif verb == Verb.STEAL:
                nodes = self.pantry_nodes if self.allow_steal else []
            elif verb == Verb.FLIP:
                nodes = [0]
            else:
                nodes = []

            verb_bucket = []
            for n in nodes:
                for c in range(self.n_colors):
                    for q in range(1, self.max_qty + 1):
                        self.actions.append((v, n, c, q))
                        verb_bucket.append(flat_idx)
                        flat_idx += 1
            self.per_verb_indices.append(verb_bucket)

        self.action_to_index: Dict[Tuple[int, int, int, int], int] = {
            action: idx for idx, action in enumerate(self.actions)
        }

        # component arrays for every flat action index
        arr = np.array(self.actions, dtype=np.int64)  # [A, 4]
        self.act_verb  = arr[:, 0]
        self.act_node  = arr[:, 1]
        self.act_color = arr[:, 2]
        self.act_qty   = arr[:, 3]

        self.t_act_verb  = torch.as_tensor( arr[:, 0] , dtype=torch.long)
        self.t_act_node  = torch.as_tensor( arr[:, 1] , dtype=torch.long)
        self.t_act_color = torch.as_tensor( arr[:, 2] , dtype=torch.long)
        self.t_act_qty   = torch.as_tensor( arr[:, 3] , dtype=torch.long)

        # add after self.act_qty = arr[:, 3]
        self.idx_pick  = np.asarray(self.per_verb_indices[Verb.PICK],  dtype=np.int64)
        self.idx_place = np.asarray(self.per_verb_indices[Verb.PLACE], dtype=np.int64)
        self.idx_flip  = np.asarray(self.per_verb_indices[Verb.FLIP],  dtype=np.int64)
        self.idx_steal = np.asarray(self.per_verb_indices[Verb.STEAL], dtype=np.int64)


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

        pan_flat = obs[self.sl_pan]
        pan = pan_flat.reshape(self.n_pantries, self.n_colors).copy()
        pan_sum = pan.sum(axis=1)
        pan_room = np.clip(self.pantry_cap - pan_sum, a_min=0.0, a_max=None)

        pick_flat = obs[self.sl_pick]
        pick = pick_flat.reshape(self.n_pickups, self.n_colors).copy()
        pick_sum = pick.sum(axis=1)

        nest_counts = obs[self.sl_nest]
        nest_blue = float(nest_counts[0])

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
    def legal_action_mask(self, obs: np.ndarray) -> np.ndarray:
        ctx = self._ctx(obs)
        A = len(self.actions)
        mask = np.zeros(A, dtype=bool)

        # --- PICK ---
        if self.idx_pick.size:
            nodes  = self.act_node[self.idx_pick]
            colors = self.act_color[self.idx_pick]
            qty    = self.act_qty[self.idx_pick].astype(np.float32)

            loc      = self.pickup_idx[nodes]
            stock_t  = ctx.pick_sum[loc]                 # total at pickup
            stock_c  = ctx.pick[loc, colors]             # chosen color at pickup
            cap      = ctx.cap_left
            limit    = np.minimum.reduce([stock_c, np.full_like(stock_c, cap), np.full_like(stock_c, self.max_qty, dtype=np.float32)])

            valid = (cap > 0.0) & (stock_t > 0.0) & (stock_c > 0.0) & (qty <= limit)
            mask[self.idx_pick] = valid

        # --- PLACE (pantries + nest) ---
        if self.idx_place.size:
            nodes  = self.act_node[self.idx_place]
            colors = self.act_color[self.idx_place]
            qty    = self.act_qty[self.idx_place].astype(np.float32)

            inv_c = ctx.inv_b[colors]                    # inventory of chosen color

            # Nest placements
            to_nest = (nodes == self.nest_blue_node)
            if np.any(to_nest):
                limit_n = np.minimum.reduce([
                    np.full(np.count_nonzero(to_nest), ctx.nest_room, dtype=np.float32),
                    inv_c[to_nest],
                    np.full(np.count_nonzero(to_nest), self.max_qty, dtype=np.float32),
                ])
                valid_n = (inv_c[to_nest] > 0.0) & (ctx.nest_room > 0.0) & (qty[to_nest] <= limit_n)
                m = mask[self.idx_place]
                m[to_nest] = valid_n
                mask[self.idx_place] = m  # writeback

            # Pantry placements
            to_pan = ~to_nest
            if np.any(to_pan):
                loc      = self.pantry_idx[nodes[to_pan]]
                room     = ctx.pan_room[loc]
                limit_p  = np.minimum.reduce([room, inv_c[to_pan], np.full_like(room, self.max_qty, dtype=np.float32)])
                valid_p  = (inv_c[to_pan] > 0.0) & (room > 0.0) & (qty[to_pan] <= limit_p)
                m = mask[self.idx_place]
                m[to_pan] = valid_p
                mask[self.idx_place] = m

        # --- FLIP ---
        if self.idx_flip.size and self.can_flip:
            nodes  = self.act_node[self.idx_flip]
            colors = self.act_color[self.idx_flip]
            qty    = self.act_qty[self.idx_flip].astype(np.float32)

            other_c   = 1 - colors
            stock_o   = ctx.inv_b[other_c]
            limit_f   = np.minimum(stock_o, np.full_like(stock_o, self.max_qty, dtype=np.float32))
            valid_f   = (nodes == 0) & (stock_o > 0.0) & (qty <= limit_f)
            mask[self.idx_flip] = valid_f

        # --- STEAL ---
        if self.idx_steal.size and self.allow_steal:
            nodes  = self.act_node[self.idx_steal]
            colors = self.act_color[self.idx_steal]
            qty    = self.act_qty[self.idx_steal].astype(np.float32)

            cap    = ctx.cap_left
            loc    = self.pantry_idx[nodes]
            stock  = ctx.pan[loc, colors]
            limit_s = np.minimum.reduce([stock, np.full_like(stock, cap), np.full_like(stock, self.max_qty, dtype=np.float32)])
            valid_s = (cap > 0.0) & (stock > 0.0) & (qty <= limit_s)
            mask[self.idx_steal] = valid_s

        return mask

    def legal_actions(self, obs: np.ndarray) -> List[int]:
        return np.flatnonzero(self.legal_action_mask(obs)).tolist()

