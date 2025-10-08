from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch as th
from torch.distributions import Categorical

from stable_baselines3.common.policies import ActorCriticPolicy

from robot import Verb

BIG_NEG = -1e9
COLOR_BLUE = 0
COLOR_YELLOW = 1


@dataclass
class MaskContext:
    blue_node: th.Tensor
    inv_b: th.Tensor
    inv_sum: th.Tensor
    cap_left: th.Tensor
    pan: th.Tensor
    pick: th.Tensor
    pan_room: th.Tensor
    pan_sum: th.Tensor
    pick_sum: th.Tensor
    nest_blue: th.Tensor


class MaskedMultiCatPolicy(ActorCriticPolicy):
    """Branched masked policy that samples actions sequentially with verb-dependent masks."""

    def __init__(self, *args, **kwargs):
        self.mask_cfg: Dict = kwargs.pop("mask_cfg")
        super().__init__(*args, **kwargs)

        nvec = np.array(self.action_space.nvec, dtype=np.int64)
        self.register_buffer("nvec", th.as_tensor(nvec, dtype=th.long))
        self.param_dim = int(nvec.sum())
        assert self.action_net.out_features == self.param_dim
        self.n_verbs = int(nvec[0])
        self.n_nodes = int(nvec[1])
        self.n_colors = int(nvec[2])
        self.max_qtyp1 = int(nvec[3])

        # Precompute slices for each action head within concatenated logits
        head_offsets = []
        offset = 0
        for size in nvec:
            head_offsets.append((int(offset), int(offset + size)))
            offset += int(size)
        self.slices = head_offsets  # [(start,end) for verb,node,color,qty]

        # Environment constants
        self.n_pantries = int(self.mask_cfg.get("n_pantries", 0))
        self.n_pickups = int(self.mask_cfg.get("n_pickups", 0))
        self.capacity = float(self.mask_cfg.get("capacity", 999))
        self.pantry_cap = float(self.mask_cfg.get("pantry_cap", 1e9))
        self.allow_steal = bool(self.mask_cfg.get("allow_steal", True))
        self.can_flip = bool(self.mask_cfg.get("can_flip", True))
        self.nest_blue = int(self.mask_cfg.get("nest_blue", -1))
        self.nest_cap = float(self.mask_cfg.get("nest_cap_blue", 0.0))

        # Lookup buffers
        pantry_idx = np.asarray(self.mask_cfg.get("pantry_idx", []), dtype=np.int64)
        pickup_idx = np.asarray(self.mask_cfg.get("pickup_idx", []), dtype=np.int64)
        pantry_nodes = np.asarray(self.mask_cfg.get("pantry_nodes", []), dtype=np.int64)
        pickup_nodes = np.asarray(self.mask_cfg.get("pickup_nodes", []), dtype=np.int64)
        self.register_buffer("pantry_idx", th.as_tensor(pantry_idx, dtype=th.long))
        self.register_buffer("pickup_idx", th.as_tensor(pickup_idx, dtype=th.long))
        self.register_buffer("pantry_nodes", th.as_tensor(pantry_nodes, dtype=th.long))
        self.register_buffer("pickup_nodes", th.as_tensor(pickup_nodes, dtype=th.long))

        # Observation slices
        i = 0
        self.sl_t_by = (i, i + 3)
        i += 3
        self.sl_inv_b = (i, i + self.n_colors)
        i += self.n_colors
        self.sl_inv_y = (i, i + self.n_colors)
        i += self.n_colors
        self.sl_pan = (i, i + self.n_colors * self.n_pantries)
        i += self.n_colors * self.n_pantries
        self.sl_pick = (i, i + self.n_colors * self.n_pickups)
        i += self.n_colors * self.n_pickups
        self.sl_nest = (i, i + 2)

    # ---- context helpers -------------------------------------------------
    def _build_context(self, obs: th.Tensor) -> MaskContext:
        B = obs.shape[0]
        device = obs.device

        blue_node = obs[:, self.sl_t_by[0] + 1].long()
        inv_b = obs[:, self.sl_inv_b[0]: self.sl_inv_b[1]]
        inv_sum = inv_b.sum(dim=1)
        cap_left = (th.tensor(self.capacity, device=device) - inv_sum).clamp_min(0)

        if self.n_pantries > 0:
            pan_flat = obs[:, self.sl_pan[0]: self.sl_pan[1]]
            pan = pan_flat.view(B, self.n_pantries, self.n_colors)
            pan_sum = pan.sum(dim=2)
            pan_room = (th.tensor(self.pantry_cap, device=device) - pan_sum).clamp_min(0)
        else:
            pan = th.zeros((B, 0, self.n_colors), device=device, dtype=obs.dtype)
            pan_sum = th.zeros((B, 0), device=device, dtype=obs.dtype)
            pan_room = th.zeros((B, 0), device=device, dtype=obs.dtype)

        if self.n_pickups > 0:
            pick_flat = obs[:, self.sl_pick[0]: self.sl_pick[1]]
            pick = pick_flat.view(B, self.n_pickups, self.n_colors)
            pick_sum = pick.sum(dim=2)
        else:
            pick = th.zeros((B, 0, self.n_colors), device=device, dtype=obs.dtype)
            pick_sum = th.zeros((B, 0), device=device, dtype=obs.dtype)

        if self.sl_nest[1] > self.sl_nest[0]:
            nest_counts = obs[:, self.sl_nest[0]: self.sl_nest[1]]
            nest_blue = nest_counts[:, 0]
        else:
            nest_blue = th.zeros(B, device=device, dtype=obs.dtype)

        return MaskContext(
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
        )

    def _nest_capacity_left(self, ctx: MaskContext) -> th.Tensor:
        if self.nest_blue < 0 or self.nest_cap <= 0:
            return th.zeros_like(ctx.nest_blue)
        cap = th.as_tensor(self.nest_cap, device=ctx.nest_blue.device, dtype=ctx.nest_blue.dtype)
        return (cap - ctx.nest_blue).clamp_min(0)

    def _mask_verb(self, ctx: MaskContext) -> th.Tensor:
        B = ctx.blue_node.size(0)
        device = ctx.blue_node.device
        mask = th.zeros((B, self.n_verbs), dtype=th.bool, device=device)

        if Verb.PICK < self.n_verbs and self.n_pickups > 0:
            has_pick = ((ctx.pick_sum > 0) & (ctx.cap_left.view(B, 1) > 0)).any(dim=1)
            mask[:, Verb.PICK] = has_pick

        if Verb.PLACE < self.n_verbs:
            has_inventory = ctx.inv_sum > 0
            pantry_room_any = (self.n_pantries > 0) and (ctx.pan_room > 0).any(dim=1)
            nest_room = self._nest_capacity_left(ctx) > 0
            mask[:, Verb.PLACE] = has_inventory & (pantry_room_any | nest_room)

        if Verb.FLIP < self.n_verbs and self.can_flip:
            has_source = (ctx.inv_sum > 0) & ((ctx.inv_sum.unsqueeze(1) - ctx.inv_b) > 0).any(dim=1)
            mask[:, Verb.FLIP] = has_source

        if Verb.STEAL < self.n_verbs and self.allow_steal and self.n_pantries > 0:
            opp_stock = ctx.pan[:, :, COLOR_YELLOW] if ctx.pan.numel() > 0 else th.zeros((B, 0), device=device)
            has_steal = (opp_stock > 0).any(dim=1) & (ctx.cap_left > 0)
            mask[:, Verb.STEAL] = has_steal

        dead = (~mask).all(dim=1)
        if bool(dead.any()):
            raise RuntimeError("MaskedMultiCatPolicy: no valid verbs available for at least one batch element.")

        return mask

    def _mask_node(self, ctx: MaskContext, verb: th.Tensor) -> th.Tensor:
        B = verb.size(0)
        device = verb.device
        mask = th.zeros((B, self.n_nodes), dtype=th.bool, device=device)
        current_node = ctx.blue_node
        pick_mask = verb == Verb.PICK
        if pick_mask.any():
            submask = mask[pick_mask]
            if submask.size(0) > 0 and self.n_pickups > 0:
                valid = (ctx.pick_sum > 0) & (ctx.cap_left.view(B, 1) > 0)
                submask[:, self.pickup_nodes] = valid[pick_mask]
            mask[pick_mask] = submask

        place_mask = verb == Verb.PLACE
        if place_mask.any():
            submask = mask[place_mask]
            if submask.size(0) > 0:
                if self.n_pantries > 0:
                    room = (ctx.pan_room > 0) & (ctx.inv_sum.view(B, 1) > 0)
                    submask[:, self.pantry_nodes] |= room[place_mask]
                if self.nest_blue >= 0:
                    nest_room = self._nest_capacity_left(ctx) > 0
                    submask[:, self.nest_blue] |= ((ctx.inv_sum > 0) & nest_room)[place_mask]
            mask[place_mask] = submask

        flip_mask = verb == Verb.FLIP
        if flip_mask.any():
            mask[flip_mask, current_node[flip_mask]] = True

        steal_mask = verb == Verb.STEAL
        if steal_mask.any():
            submask = mask[steal_mask]
            if submask.size(0) > 0 and self.n_pantries > 0:
                opp_stock = ctx.pan[:, :, COLOR_YELLOW] if ctx.pan.numel() > 0 else th.zeros((B, 0), device=device)
                steal_room = (opp_stock > 0) & (ctx.cap_left.view(B, 1) > 0)
                submask[:, self.pantry_nodes] |= steal_room[steal_mask]
            mask[steal_mask] = submask

        empty = (~mask).all(dim=1)
        if bool(empty.any()):
            raise RuntimeError("MaskedMultiCatPolicy: verb allowed but no valid target nodes.")

        return mask

    def _mask_color(self, ctx: MaskContext, verb: th.Tensor, node: th.Tensor) -> th.Tensor:
        B = verb.size(0)
        device = verb.device
        mask = th.zeros((B, self.n_colors), dtype=th.bool, device=device)

        pickup_local = self.pickup_idx[node] if self.pickup_idx.numel() else th.full_like(node, -1)
        pantry_local = self.pantry_idx[node] if self.pantry_idx.numel() else th.full_like(node, -1)

        pick_mask = verb == Verb.PICK
        if pick_mask.any() and self.n_pickups > 0:
            valid_rows = pick_mask & (pickup_local >= 0)
            if valid_rows.any():
                local = pickup_local[valid_rows]
                stocks = ctx.pick[valid_rows, :, :].gather(1, local.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_colors)).squeeze(1)
                mask[valid_rows] = (stocks > 0) & (ctx.cap_left[valid_rows].unsqueeze(1) > 0)

        place_mask = verb == Verb.PLACE
        if place_mask.any():
            valid_pantry = place_mask & (pantry_local >= 0)
            if valid_pantry.any():
                local = pantry_local[valid_pantry]
                room = ctx.pan_room[valid_pantry, :].gather(1, local.unsqueeze(1)).squeeze(1)
                inv = ctx.inv_b[valid_pantry]
                mask[valid_pantry] = (inv > 0) & (room.unsqueeze(1) > 0)
            valid_nest = place_mask & (node == self.nest_blue)
            if valid_nest.any():
                inv = ctx.inv_b[valid_nest]
                nest_room = self._nest_capacity_left(ctx)[valid_nest].unsqueeze(1)
                mask[valid_nest] |= (inv > 0) & (nest_room > 0)

        flip_mask = verb == Verb.FLIP
        if flip_mask.any():
            inv_sum = ctx.inv_sum.view(B, 1)
            mask[flip_mask] = (inv_sum - ctx.inv_b)[flip_mask] > 0

        steal_mask = verb == Verb.STEAL
        if steal_mask.any() and self.n_pantries > 0:
            valid_rows = steal_mask & (pantry_local >= 0)
            if valid_rows.any():
                local = pantry_local[valid_rows]
                stocks = ctx.pan[valid_rows, :, :].gather(1, local.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_colors)).squeeze(1)
                mask[valid_rows] = (stocks > 0) & (ctx.cap_left[valid_rows].unsqueeze(1) > 0)

        empty = (~mask).all(dim=1)
        if bool(empty.any()):
            raise RuntimeError("MaskedMultiCatPolicy: verb/node combination has no valid colors.")

        return mask

    def _mask_qty(self, ctx: MaskContext, verb: th.Tensor, node: th.Tensor, color: th.Tensor) -> th.Tensor:
        B = verb.size(0)
        device = verb.device
        mask = th.zeros((B, self.max_qtyp1), dtype=th.bool, device=device)
        idx = th.arange(self.max_qtyp1, device=device).view(1, -1)

        pickup_local = self.pickup_idx[node] if self.pickup_idx.numel() else th.full_like(node, -1)
        pantry_local = self.pantry_idx[node] if self.pantry_idx.numel() else th.full_like(node, -1)

        pick_mask = verb == Verb.PICK
        if pick_mask.any() and self.n_pickups > 0:
            valid_rows = pick_mask & (pickup_local >= 0)
            if valid_rows.any():
                local = pickup_local[valid_rows]
                stocks = ctx.pick[valid_rows, :, :].gather(1, local.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_colors)).squeeze(1)
                stock_color = stocks[th.arange(stocks.size(0), device=device), color[valid_rows]]
                max_stock = th.floor(stock_color).long()
                cap = th.floor(ctx.cap_left[valid_rows]).long()
                limit = th.minimum(max_stock, cap).clamp(min=0, max=self.max_qtyp1 - 1)
                mask_vals = (idx >= 1) & (idx <= limit.unsqueeze(1))
                mask[valid_rows] = mask_vals

        place_mask = verb == Verb.PLACE
        if place_mask.any():
            valid_pantry = place_mask & (pantry_local >= 0)
            if valid_pantry.any():
                local = pantry_local[valid_pantry]
                room = ctx.pan_room[valid_pantry, :].gather(1, local.unsqueeze(1)).squeeze(1)
                inv = ctx.inv_b[valid_pantry, color[valid_pantry]]
                limit = th.minimum(th.floor(room).long(), th.floor(inv).long()).clamp(min=0, max=self.max_qtyp1 - 1)
                mask_vals = (idx >= 1) & (idx <= limit.unsqueeze(1))
                mask[valid_pantry] = mask_vals
            valid_nest = place_mask & (node == self.nest_blue)
            if valid_nest.any():
                inv_rows = ctx.inv_b[valid_nest]
                selected = color[valid_nest].unsqueeze(1)
                inv = inv_rows.gather(1, selected).squeeze(1)
                nest_cap_left = self._nest_capacity_left(ctx)[valid_nest]
                limit = th.minimum(th.floor(inv).long(), th.floor(nest_cap_left).long()).clamp(min=0, max=self.max_qtyp1 - 1)
                mask_vals = (idx >= 1) & (idx <= limit.unsqueeze(1))
                mask[valid_nest] = mask_vals

        flip_mask = verb == Verb.FLIP
        if flip_mask.any():
            other_color = 1 - color[flip_mask]
            inv_other = ctx.inv_b[flip_mask, other_color]
            limit = th.floor(inv_other).long().clamp(min=0, max=self.max_qtyp1 - 1)
            mask_vals = (idx >= 1) & (idx <= limit.unsqueeze(1))
            mask[flip_mask] = mask_vals

        steal_mask = verb == Verb.STEAL
        if steal_mask.any() and self.n_pantries > 0:
            valid_rows = steal_mask & (pantry_local >= 0)
            if valid_rows.any():
                local = pantry_local[valid_rows]
                stocks = ctx.pan[valid_rows, :, :].gather(1, local.unsqueeze(1).unsqueeze(2).expand(-1, 1, self.n_colors)).squeeze(1)
                stock_color = stocks[th.arange(stocks.size(0), device=device), color[valid_rows]]
                max_stock = th.floor(stock_color).long()
                cap = th.floor(ctx.cap_left[valid_rows]).long()
                limit = th.minimum(max_stock, cap).clamp(min=0, max=self.max_qtyp1 - 1)
                mask_vals = (idx >= 1) & (idx <= limit.unsqueeze(1))
                mask[valid_rows] = mask_vals

        empty = (~mask).all(dim=1)
        if bool(empty.any()):
            raise RuntimeError("MaskedMultiCatPolicy: no valid quantities for selected verb/node/color.")

        return mask

    # ---- masking utilities ----------------------------------------------
    @staticmethod
    def _apply_mask(logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask = mask.bool()
        masked = th.where(mask, logits, th.full_like(logits, BIG_NEG))
        dead = (~mask).all(dim=1, keepdim=True)
        if dead.any():
            masked = masked.clone()
            fallback = logits.max(dim=1, keepdim=True).values
            masked[dead] = fallback[dead]
        return masked

    def _sample_head(self, logits: th.Tensor, mask: th.Tensor, deterministic: bool) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        masked_logits = self._apply_mask(logits, mask)
        dist = Categorical(logits=masked_logits)
        if deterministic:
            actions = masked_logits.argmax(dim=1)
        else:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions.long(), log_prob, entropy

    # ---- sampling / evaluation -------------------------------------------
    def _latent_pi_vf(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def _split_logits(self, latent_pi: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        logits_all = self.action_net(latent_pi)
        logits_heads = [logits_all[:, s:e] for (s, e) in self.slices]
        return tuple(logits_heads)  # verb, node, color, qty

    def _sample_actions(self, obs: th.Tensor, latent_pi: th.Tensor, deterministic: bool = False):
        verb_logits, node_logits, color_logits, qty_logits = self._split_logits(latent_pi)
        ctx = self._build_context(obs)

        verb_mask = self._mask_verb(ctx)
        verb, logp_verb, ent_verb = self._sample_head(verb_logits, verb_mask, deterministic)

        B = obs.size(0)
        device = obs.device

        node = ctx.blue_node.clone()
        logp_node = th.zeros(B, device=device)
        ent_node = th.zeros(B, device=device)
        node_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.STEAL)
        if node_required.any():
            node_mask = self._mask_node(ctx, verb)
            sel_node, lp_node, en_node = self._sample_head(node_logits[node_required], node_mask[node_required], deterministic)
            node[node_required] = sel_node
            logp_node[node_required] = lp_node
            ent_node[node_required] = en_node

        color = th.zeros(B, dtype=th.long, device=device)
        logp_color = th.zeros(B, device=device)
        ent_color = th.zeros(B, device=device)
        color_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if color_required.any():
            color_mask = self._mask_color(ctx, verb, node)
            sel_color, lp_color, en_color = self._sample_head(color_logits[color_required], color_mask[color_required], deterministic)
            color[color_required] = sel_color
            logp_color[color_required] = lp_color
            ent_color[color_required] = en_color

        qty = th.zeros(B, dtype=th.long, device=device)
        logp_qty = th.zeros(B, device=device)
        ent_qty = th.zeros(B, device=device)
        qty_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if qty_required.any():
            qty_mask = self._mask_qty(ctx, verb, node, color)
            sel_qty, lp_qty, en_qty = self._sample_head(qty_logits[qty_required], qty_mask[qty_required], deterministic)
            qty[qty_required] = sel_qty
            logp_qty[qty_required] = lp_qty
            ent_qty[qty_required] = en_qty

        actions = th.stack([verb, node, color, qty], dim=1)
        log_prob = logp_verb + logp_node + logp_color + logp_qty
        entropy = ent_verb + ent_node + ent_color + ent_qty
        return actions, log_prob, entropy

    def _evaluate(self, obs: th.Tensor, latent_pi: th.Tensor, actions: th.Tensor):
        verb_logits, node_logits, color_logits, qty_logits = self._split_logits(latent_pi)
        ctx = self._build_context(obs)

        verb = actions[:, 0].long()
        node = actions[:, 1].long()
        color = actions[:, 2].long()
        qty = actions[:, 3].long()

        device = obs.device
        B = obs.size(0)

        verb_mask = self._mask_verb(ctx)
        masked_verb = self._apply_mask(verb_logits, verb_mask)
        verb_dist = Categorical(logits=masked_verb)
        logp_verb = verb_dist.log_prob(verb)
        ent_verb = verb_dist.entropy()

        logp_node = th.zeros(B, device=device)
        ent_node = th.zeros(B, device=device)
        node_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.STEAL)
        if node_required.any():
            node_mask = self._mask_node(ctx, verb)
            masked = self._apply_mask(node_logits[node_required], node_mask[node_required])
            dist = Categorical(logits=masked)
            logp_node[node_required] = dist.log_prob(node[node_required])
            ent_node[node_required] = dist.entropy()

        logp_color = th.zeros(B, device=device)
        ent_color = th.zeros(B, device=device)
        color_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if color_required.any():
            color_mask = self._mask_color(ctx, verb, node)
            masked = self._apply_mask(color_logits[color_required], color_mask[color_required])
            dist = Categorical(logits=masked)
            logp_color[color_required] = dist.log_prob(color[color_required])
            ent_color[color_required] = dist.entropy()

        logp_qty = th.zeros(B, device=device)
        ent_qty = th.zeros(B, device=device)
        qty_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if qty_required.any():
            qty_mask = self._mask_qty(ctx, verb, node, color)
            masked = self._apply_mask(qty_logits[qty_required], qty_mask[qty_required])
            dist = Categorical(logits=masked)
            logp_qty[qty_required] = dist.log_prob(qty[qty_required])
            ent_qty[qty_required] = dist.entropy()

        log_prob = logp_verb + logp_node + logp_color + logp_qty
        entropy = ent_verb + ent_node + ent_color + ent_qty
        return log_prob, entropy

    # ---- SB3 overrides ---------------------------------------------------
    def forward(self, obs: th.Tensor, deterministic: bool = False):  # type: ignore[override]
        latent_pi, latent_vf = self._latent_pi_vf(obs)
        actions, log_prob, _ = self._sample_actions(obs, latent_pi, deterministic)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):  # type: ignore[override]
        latent_pi, latent_vf = self._latent_pi_vf(obs)
        log_prob, entropy = self._evaluate(obs, latent_pi, actions)
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):  # type: ignore[override]
        latent_pi, _ = self._latent_pi_vf(obs)
        policy = self

        class BranchedDistribution:
            def __init__(self, obs: th.Tensor, latent_pi: th.Tensor):
                self._obs = obs
                self._latent_pi = latent_pi

            def get_actions(self, deterministic: bool = False):
                actions, _, _ = policy._sample_actions(self._obs, self._latent_pi, deterministic)
                return actions

            def log_prob(self, actions: th.Tensor):
                log_prob, _ = policy._evaluate(self._obs, self._latent_pi, actions)
                return log_prob

            def entropy(self):
                # entropy for deterministic trajectories is only used for logging;
                # compute using freshly sampled actions.
                actions, _, entropy = policy._sample_actions(self._obs, self._latent_pi, deterministic=False)
                return entropy

        return BranchedDistribution(obs, latent_pi)
