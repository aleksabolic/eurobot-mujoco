from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import os, sys
import numpy as np
import torch as th
from torch.distributions import Categorical
from torch.utils.cpp_extension import load

from stable_baselines3.common.policies import ActorCriticPolicy

from robot import Verb

BIG_NEG = -1e9
COLOR_BLUE = 0
COLOR_YELLOW = 1

_MASK_EXT = None

def _load_mask_ext():
    global _MASK_EXT
    if _MASK_EXT is None:
        src_path = Path(__file__).resolve().with_name("mask_ext.cpp")
        is_win = os.name == "nt"
        cflags = ["/O2","/openmp"] if is_win else ["-O3","-fopenmp"]
        ldflags = [] if is_win else ["-fopenmp"]
        _MASK_EXT = load(name="mask_ext", sources=[str(src_path)],
                 extra_cflags=cflags, extra_ldflags=ldflags, verbose=False)
    return _MASK_EXT


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
        self._capacity_int = int(self.capacity)
        self.pantry_cap = float(self.mask_cfg.get("pantry_cap", 1e9))
        self._pantry_cap_int = int(self.pantry_cap)
        self.allow_steal = bool(self.mask_cfg.get("allow_steal", True))
        self.can_flip = bool(self.mask_cfg.get("can_flip", True))
        self.nest_blue = int(self.mask_cfg.get("nest_blue", -1))
        self.nest_cap = float(self.mask_cfg.get("nest_cap_blue", 0.0))
        self._nest_cap_int = int(self.nest_cap)
        self.big_neg = float(self.mask_cfg.get("big_neg", BIG_NEG))
        self.flip_node_all = bool(self.mask_cfg.get("flip_node_all", True))

        # Lookup buffers
        pantry_idx = np.asarray(self.mask_cfg.get("pantry_idx", []), dtype=np.int64)
        pickup_idx = np.asarray(self.mask_cfg.get("pickup_idx", []), dtype=np.int64)
        pantry_nodes = np.asarray(self.mask_cfg.get("pantry_nodes", []), dtype=np.int64)
        pickup_nodes = np.asarray(self.mask_cfg.get("pickup_nodes", []), dtype=np.int64)
        self.register_buffer("pantry_idx", th.as_tensor(pantry_idx, dtype=th.long))
        self.register_buffer("pickup_idx", th.as_tensor(pickup_idx, dtype=th.long))
        self.register_buffer("pantry_nodes", th.as_tensor(pantry_nodes, dtype=th.long))
        self.register_buffer("pickup_nodes", th.as_tensor(pickup_nodes, dtype=th.long))

        self._pantry_idx_cpu = self.pantry_idx.detach().cpu()
        self._pickup_idx_cpu = self.pickup_idx.detach().cpu()
        self._pantry_nodes_cpu = self.pantry_nodes.detach().cpu()
        self._pickup_nodes_cpu = self.pickup_nodes.detach().cpu()
        self._mask_ext = _load_mask_ext()

        # Observation slices mirror EurobotDiscreteEnv._obs layout.
        i = 0
        self.sl_node_blue = (i, i + self.n_nodes)
        i += self.n_nodes
        self.sl_node_yellow = (i, i + self.n_nodes)
        i += self.n_nodes
        self.sl_inv_b = (i, i + self.n_colors)
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

        blue_slice = obs[:, self.sl_node_blue[0]: self.sl_node_blue[1]]
        blue_node = th.argmax(blue_slice, dim=1).long()
        inv_b = obs[:, self.sl_inv_b[0]: self.sl_inv_b[1]]
        inv_sum = inv_b.sum(dim=1)
        cap_left = (th.tensor(self.capacity, device=device) - inv_sum).clamp_min(0)

        # pantries
        pan_flat = obs[:, self.sl_pan[0]: self.sl_pan[1]]
        pan = pan_flat.view(B, self.n_pantries, self.n_colors)
        pan_sum = pan.sum(dim=2)
        pan_room = (th.tensor(self.pantry_cap, device=device) - pan_sum).clamp_min(0)

        # pickups
        pick_flat = obs[:, self.sl_pick[0]: self.sl_pick[1]]
        pick = pick_flat.view(B, self.n_pickups, self.n_colors)
        pick_sum = pick.sum(dim=2)

        # nest
        nest_counts = obs[:, self.sl_nest[0]: self.sl_nest[1]]
        nest_blue = nest_counts[:, 0]

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

    def _context_to_cpu(self, ctx: MaskContext) -> MaskContext:
        return MaskContext(
            blue_node=ctx.blue_node.detach().to("cpu", non_blocking=False).contiguous(),
            inv_b=ctx.inv_b.detach().to("cpu", non_blocking=False).contiguous(),
            inv_sum=ctx.inv_sum.detach().to("cpu", non_blocking=False).contiguous(),
            cap_left=ctx.cap_left.detach().to("cpu", non_blocking=False).contiguous(),
            pan=ctx.pan.detach().to("cpu", non_blocking=False).contiguous(),
            pick=ctx.pick.detach().to("cpu", non_blocking=False).contiguous(),
            pan_room=ctx.pan_room.detach().to("cpu", non_blocking=False).contiguous(),
            pan_sum=ctx.pan_sum.detach().to("cpu", non_blocking=False).contiguous(),
            pick_sum=ctx.pick_sum.detach().to("cpu", non_blocking=False).contiguous(),
            nest_blue=ctx.nest_blue.detach().to("cpu", non_blocking=False).contiguous(),
        )

    def _mask_ext_call(
        self,
        ctx_cpu: MaskContext,
        verb_logits_cpu: th.Tensor,
        node_logits_cpu: th.Tensor,
        color_logits_cpu: th.Tensor,
        qty_logits_cpu: th.Tensor,
        verb_sel_cpu: Optional[th.Tensor] = None,
        node_sel_cpu: Optional[th.Tensor] = None,
        color_sel_cpu: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        verb_mask, node_mask, color_mask, qty_mask, *_ = self._mask_ext.build_masks_and_apply_logits(
            verb_logits_cpu,
            node_logits_cpu,
            color_logits_cpu,
            qty_logits_cpu,
            ctx_cpu.blue_node,
            ctx_cpu.inv_b,
            ctx_cpu.inv_sum,
            ctx_cpu.cap_left,
            ctx_cpu.pan,
            ctx_cpu.pick,
            ctx_cpu.pan_room,
            ctx_cpu.pan_sum,
            ctx_cpu.pick_sum,
            ctx_cpu.nest_blue,
            self._pantry_idx_cpu,
            self._pickup_idx_cpu,
            self._pantry_nodes_cpu,
            self._pickup_nodes_cpu,
            self.n_verbs,
            self.n_nodes,
            self.n_colors,
            self.max_qtyp1,
            self._capacity_int,
            self._pantry_cap_int,
            self._nest_cap_int,
            self.nest_blue,
            self.allow_steal,
            self.can_flip,
            self.big_neg,
            False,
            self.flip_node_all,
            verb_sel_cpu,
            node_sel_cpu,
            color_sel_cpu,
        )
        return verb_mask, node_mask, color_mask, qty_mask

    # ---- masking utilities ----------------------------------------------
    def _apply_mask(self, logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask = mask.bool()
        masked = th.where(mask, logits, th.full_like(logits, self.big_neg))
        dead = (~mask).all(dim=1, keepdim=True)
        if dead.any():
            masked = masked.clone()
            fallback = logits.max(dim=1, keepdim=True).values
            masked[dead] = fallback[dead]
        return masked
    
    def _sample_head(self, logits: th.Tensor, mask: th.Tensor, deterministic):
        masked_logits = self._apply_mask(logits, mask)
        if deterministic:
            actions = masked_logits.argmax(dim=1)
            logps = th.log_softmax(masked_logits, dim=1)
            logp = logps.gather(1, actions[:,None]).squeeze(1)
            ent = th.zeros_like(logp)  # if you only log entropy elsewhere
            return actions, logp, ent
        u = th.rand_like(masked_logits)
        g = -th.log(-th.log(u.clamp_min(1e-6)))
        actions = (masked_logits + g).argmax(dim=1)
        logps = th.log_softmax(masked_logits, dim=1)
        logp = logps.gather(1, actions[:,None]).squeeze(1)
        ent = -(logps.exp() * logps).sum(dim=1)
        return actions, logp, ent


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
        ctx_cpu = self._context_to_cpu(ctx)

        B = obs.size(0)
        device = obs.device

        verb_logits_cpu = verb_logits.detach().to("cpu", non_blocking=False).contiguous()
        node_logits_cpu = node_logits.detach().to("cpu", non_blocking=False).contiguous()
        color_logits_cpu = color_logits.detach().to("cpu", non_blocking=False).contiguous()
        qty_logits_cpu = qty_logits.detach().to("cpu", non_blocking=False).contiguous()

        verb_mask_cpu, _, _, _ = self._mask_ext_call(
            ctx_cpu,
            verb_logits_cpu,
            node_logits_cpu,
            color_logits_cpu,
            qty_logits_cpu,
        )
        verb_mask = verb_mask_cpu.to(device=device)
        verb, logp_verb, ent_verb = self._sample_head(verb_logits, verb_mask, deterministic)

        node = ctx.blue_node.clone()
        logp_node = th.zeros(B, device=device)
        ent_node = th.zeros(B, device=device)
        node_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.STEAL)
        if node_required.any():
            verb_sel = verb.clone()
            verb_sel[~node_required] = -1
            verb_sel_cpu = verb_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            _, node_mask_cpu, _, _ = self._mask_ext_call(
                ctx_cpu,
                verb_logits_cpu,
                node_logits_cpu,
                color_logits_cpu,
                qty_logits_cpu,
                verb_sel_cpu=verb_sel_cpu,
            )
            node_mask = node_mask_cpu.to(device=device)
            sel_node, lp_node, en_node = self._sample_head(
                node_logits[node_required], node_mask[node_required], deterministic
            )
            node[node_required] = sel_node
            logp_node[node_required] = lp_node
            ent_node[node_required] = en_node

        color = th.zeros(B, dtype=th.long, device=device)
        logp_color = th.zeros(B, device=device)
        ent_color = th.zeros(B, device=device)
        color_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if color_required.any():
            verb_sel = verb.clone()
            verb_sel[~color_required] = -1
            node_sel = node.clone()
            node_sel[~color_required] = -1
            verb_sel_cpu = verb_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            node_sel_cpu = node_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            _, _, color_mask_cpu, _ = self._mask_ext_call(
                ctx_cpu,
                verb_logits_cpu,
                node_logits_cpu,
                color_logits_cpu,
                qty_logits_cpu,
                verb_sel_cpu=verb_sel_cpu,
                node_sel_cpu=node_sel_cpu,
            )
            color_mask = color_mask_cpu.to(device=device)
            sel_color, lp_color, en_color = self._sample_head(
                color_logits[color_required], color_mask[color_required], deterministic
            )
            color[color_required] = sel_color
            logp_color[color_required] = lp_color
            ent_color[color_required] = en_color

        qty = th.zeros(B, dtype=th.long, device=device)
        logp_qty = th.zeros(B, device=device)
        ent_qty = th.zeros(B, device=device)
        qty_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if qty_required.any():
            verb_sel = verb.clone()
            verb_sel[~qty_required] = -1
            node_sel = node.clone()
            node_sel[~qty_required] = -1
            color_sel = color.clone()
            color_sel[~qty_required] = -1
            verb_sel_cpu = verb_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            node_sel_cpu = node_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            color_sel_cpu = color_sel.to(th.long).to("cpu", non_blocking=False).contiguous()
            _, _, _, qty_mask_cpu = self._mask_ext_call(
                ctx_cpu,
                verb_logits_cpu,
                node_logits_cpu,
                color_logits_cpu,
                qty_logits_cpu,
                verb_sel_cpu=verb_sel_cpu,
                node_sel_cpu=node_sel_cpu,
                color_sel_cpu=color_sel_cpu,
            )
            qty_mask = qty_mask_cpu.to(device=device)
            sel_qty, lp_qty, en_qty = self._sample_head(
                qty_logits[qty_required], qty_mask[qty_required], deterministic
            )
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
        ctx_cpu = self._context_to_cpu(ctx)

        verb = actions[:, 0].long()
        node = actions[:, 1].long()
        color = actions[:, 2].long()
        qty = actions[:, 3].long()

        device = obs.device
        B = obs.size(0)

        verb_logits_cpu = verb_logits.detach().to("cpu", non_blocking=False).contiguous()
        node_logits_cpu = node_logits.detach().to("cpu", non_blocking=False).contiguous()
        color_logits_cpu = color_logits.detach().to("cpu", non_blocking=False).contiguous()
        qty_logits_cpu = qty_logits.detach().to("cpu", non_blocking=False).contiguous()

        verb_sel_cpu = verb.detach().to("cpu", non_blocking=False).contiguous()
        node_sel_cpu = node.detach().to("cpu", non_blocking=False).contiguous()
        color_sel_cpu = color.detach().to("cpu", non_blocking=False).contiguous()

        verb_mask_cpu, node_mask_cpu, color_mask_cpu, qty_mask_cpu = self._mask_ext_call(
            ctx_cpu,
            verb_logits_cpu,
            node_logits_cpu,
            color_logits_cpu,
            qty_logits_cpu,
            verb_sel_cpu=verb_sel_cpu,
            node_sel_cpu=node_sel_cpu,
            color_sel_cpu=color_sel_cpu,
        )

        verb_mask = verb_mask_cpu.to(device=device)
        masked_verb = self._apply_mask(verb_logits, verb_mask)
        verb_dist = Categorical(logits=masked_verb)
        logp_verb = verb_dist.log_prob(verb)
        ent_verb = verb_dist.entropy()

        logp_node = th.zeros(B, device=device)
        ent_node = th.zeros(B, device=device)
        node_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.STEAL)
        if node_required.any():
            node_mask = node_mask_cpu.to(device=device)
            masked = self._apply_mask(node_logits[node_required], node_mask[node_required])
            dist = Categorical(logits=masked)
            logp_node[node_required] = dist.log_prob(node[node_required])
            ent_node[node_required] = dist.entropy()

        logp_color = th.zeros(B, device=device)
        ent_color = th.zeros(B, device=device)
        color_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if color_required.any():
            color_mask = color_mask_cpu.to(device=device)
            masked = self._apply_mask(color_logits[color_required], color_mask[color_required])
            dist = Categorical(logits=masked)
            logp_color[color_required] = dist.log_prob(color[color_required])
            ent_color[color_required] = dist.entropy()

        logp_qty = th.zeros(B, device=device)
        ent_qty = th.zeros(B, device=device)
        qty_required = (verb == Verb.PICK) | (verb == Verb.PLACE) | (verb == Verb.FLIP) | (verb == Verb.STEAL)
        if qty_required.any():
            qty_mask = qty_mask_cpu.to(device=device)
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
