# src/masked_policy.py
from typing import Dict, Tuple
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from robot import Verb

BIG_NEG = -1e9  # for masking invalid logits


class MaskedMultiCatPolicy(ActorCriticPolicy):
    """
    MultiDiscrete policy that applies per-head action masks to logits before sampling.
    Provide env sizes/lookups via policy_kwargs['mask_cfg'] when constructing PPO:
      mask_cfg = {
        "n_pantries": K, "n_pickups": M, "n_nodes": n_nodes, "max_qty": max_qty,
        "capacity": capacity, "pantry_cap": pantry_cap, "allow_steal": True/False,
        "can_flip": True/False, "nest_blue": nest_node_index, "nest_yellow": nest_node_index,
        "pantry_idx": list[int] len==n_nodes (local index or -1),
        "pickup_idx": list[int] len==n_nodes (local index or -1),
      }
    Obs layout (EurobotDiscreteEnv._obs):
      [ t_left*10, blue_node, yellow_node,
        blue_inv(n_colors), yellow_inv(n_colors),
        pantries(n_colors*K), pickups(n_colors*M) ]
    """

    def __init__(self, *args, **kwargs):
        self.mask_cfg: Dict = kwargs.pop("mask_cfg")
        super().__init__(*args, **kwargs)

        # Action heads
        nvec = np.array(self.action_space.nvec, dtype=np.int64)
        self.register_buffer("nvec", th.as_tensor(nvec, dtype=th.long))
        self.param_dim = int(nvec.sum())
        assert self.action_net.out_features == self.param_dim
        self.dist: MultiCategoricalDistribution = self.action_dist  # type: ignore
        self.n_colors = int(nvec[2])

        # Head slices over concatenated logits
        cuts = np.cumsum(nvec)
        self.slices = [(int(0 if i == 0 else cuts[i - 1]), int(cuts[i])) for i in range(len(nvec))]

        # Parse/keep env constants
        K = int(self.mask_cfg["n_pantries"])
        M = int(self.mask_cfg["n_pickups"])
        self.n_nodes = int(self.mask_cfg["n_nodes"])
        self.n_pantries = K
        self.n_pickups = M
        self.max_qtyp1 = int(self.mask_cfg["max_qty"]) + 1
        self.capacity = float(self.mask_cfg.get("capacity", 999))
        self.allow_steal = bool(self.mask_cfg.get("allow_steal", True))
        self.can_flip = bool(self.mask_cfg.get("can_flip", True))
        self.pantry_cap = float(self.mask_cfg.get("pantry_cap", 1e9))
        self.nest_blue = int(self.mask_cfg.get("nest_blue", -1))
        self.nest_yellow = int(self.mask_cfg.get("nest_yellow", -1))

        # Node -> local index (-1 if not that kind)
        pantry_idx = np.asarray(self.mask_cfg["pantry_idx"], dtype=np.int64)
        pickup_idx = np.asarray(self.mask_cfg["pickup_idx"], dtype=np.int64)
        self.register_buffer("pantry_idx", th.as_tensor(pantry_idx, dtype=th.long))
        self.register_buffer("pickup_idx", th.as_tensor(pickup_idx, dtype=th.long))

        # Obs slices
        i = 0
        self.sl_t_by = (i, i + 3); i += 3
        self.sl_inv_b = (i, i + self.n_colors); i += self.n_colors
        self.sl_inv_y = (i, i + self.n_colors); i += self.n_colors
        self.sl_pan = (i, i + self.n_colors * K); i += self.n_colors * K
        self.sl_pick = (i, i + self.n_colors * M); i += self.n_colors * M

    # -------- mask helpers --------
    @staticmethod
    def _safe_gather_sum_colors(t_colors: th.Tensor, idx_b: th.Tensor, valid_b: th.Tensor) -> th.Tensor:
        """
        t_colors: (B, L, C), idx_b: (B,), valid_b: (B,)
        Returns (B,) = sum over color at chosen index, 0 if invalid.
        """
        B = t_colors.size(0)
        C = t_colors.size(-1)
        if C == 0:
            return th.zeros(B, device=t_colors.device, dtype=t_colors.dtype)
        idx_safe = th.clamp(idx_b, min=0)
        out = t_colors.gather(1, idx_safe.view(B, 1, 1).expand(-1, 1, C)).squeeze(1)
        out = out.sum(dim=1)
        return out * valid_b

    @staticmethod
    def _safe_gather_vec_colors(t_colors: th.Tensor, idx_b: th.Tensor, valid_b: th.Tensor) -> th.Tensor:
        """
        t_colors: (B, L, C), idx_b: (B,), valid_b: (B,)
        Returns (B,C) = row at index, zeros if invalid.
        """
        B = t_colors.size(0)
        C = t_colors.size(-1)
        if C == 0:
            return th.zeros((B, 0), device=t_colors.device, dtype=t_colors.dtype)
        idx_safe = th.clamp(idx_b, min=0)
        out = t_colors.gather(1, idx_safe.view(B, 1, 1).expand(-1, 1, C)).squeeze(1)
        return out * valid_b.view(B, 1)

    def _build_masks(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns masks for heads: (m_verb, m_node, m_color, m_qty)
        Shapes: (B,n_verbs), (B,n_nodes), (B,n_colors), (B,max_qty+1)
        """
        B = obs.shape[0]
        device = obs.device

        # Parse obs
        blue_node = obs[:, self.sl_t_by[0] + 1].long()  # blue node id
        inv_b = obs[:, self.sl_inv_b[0]: self.sl_inv_b[1]]  # (B,n_colors)
        pan_flat = obs[:, self.sl_pan[0]: self.sl_pan[1]]
        pick_flat = obs[:, self.sl_pick[0]: self.sl_pick[1]]

        inv_sum = inv_b.sum(dim=1)                         # (B,)
        cap_left = (self.capacity - inv_sum).clamp_min(0)  # (B,)

        # Reshape pantries/pickups
        K = self.n_pantries
        M = self.n_pickups
        pan = pan_flat.view(B, K, self.n_colors) if K > 0 else th.zeros((B, 0, self.n_colors), device=device)
        pick = pick_flat.view(B, M, self.n_colors) if M > 0 else th.zeros((B, 0, self.n_colors), device=device)
        
        # Node class at current position
        at_pantry = self.pantry_idx[blue_node]  # (B,)
        at_pickup = self.pickup_idx[blue_node]  # (B,)
        valid_pantry = at_pantry != -1
        valid_pickup = at_pickup != -1
        at_blue_nest = blue_node == self.nest_blue

        # Stock sums at current node (0 if invalid)
        pick_sum = self._safe_gather_sum_colors(pick, at_pickup, valid_pickup) if M > 0 else th.zeros(B, device=device)
        if K > 0:
            pan_here = self._safe_gather_vec_colors(pan, at_pantry, valid_pantry)
            pan_sum = pan_here.sum(dim=1)
            pan_cap = pan_sum.new_full((B,), float(self.pantry_cap))
            pan_room = (pan_cap - pan_sum).clamp_min(0)
        else:
            pan_sum = th.zeros(B, device=device)
            pan_here = th.zeros((B, self.n_colors), device=device)
            pan_room = th.zeros(B, device=device)

        # Verb mask (indices follow robot.Verb enum)
        n_verbs = int(self.nvec[0].item())
        m_verb = th.ones((B, n_verbs), dtype=th.bool, device=device)
        # PICK requires valid pickup, stock, capacity
        if Verb.PICK < n_verbs:
            m_verb[:, Verb.PICK] &= valid_pickup & (pick_sum > 0) & (cap_left > 0)
        # PLACE requires inventory
        if Verb.PLACE < n_verbs:
            allow_place_here = valid_pantry & (pan_room > 0)
            if self.nest_blue >= 0:
                allow_place = allow_place_here | at_blue_nest
            else:
                allow_place = allow_place_here | th.ones_like(allow_place_here, dtype=th.bool)
            m_verb[:, Verb.PLACE] &= (inv_sum > 0) & allow_place
        # FLIP requires can_flip and inventory
        if Verb.FLIP < n_verbs:
            if not self.can_flip:
                m_verb[:, Verb.FLIP] = False
            else:
                m_verb[:, Verb.FLIP] &= (inv_sum > 0)
        # WAIT allowed by default
        # STEAL requires allow_steal, valid pantry, any stock
        if Verb.STEAL < n_verbs:
            if not self.allow_steal:
                m_verb[:, Verb.STEAL] = False
            else:
                m_verb[:, Verb.STEAL] &= valid_pantry & (pan_sum > 0)

        # Node mask: allow all nodes; environment penalizes idle moves separately
        m_node = th.ones((B, self.n_nodes), dtype=th.bool, device=device)


        # Color mask:
        #  - PICK: only colors with stock>0 at current pickup
        #  - PLACE: only colors present in inventory
        #  - FLIP: allow target colors that can be produced via flipping existing stock
        m_color = th.ones((B, self.n_colors), dtype=th.bool, device=device)
        if M > 0:
            pick_here = self._safe_gather_vec_colors(pick, at_pickup, valid_pickup)
            m_color_pick = (pick_here > 0)
        else:
            m_color_pick = m_color.clone()
        m_color_place = (inv_b > 0)
        # Intersection keeps it safe for both verbs without making all-false
        m_color = m_color & (m_color_pick | m_color_place)
        # When flipping is legal, permit colors with convertible stock (any non-zero other color)
        allow_flip_head = Verb.FLIP < n_verbs and self.can_flip
        if allow_flip_head:
            flip_enabled = m_verb[:, Verb.FLIP]
            place_enabled = m_verb[:, Verb.PLACE] if Verb.PLACE < n_verbs else th.zeros_like(flip_enabled, dtype=th.bool)
            if flip_enabled.any():
                inv_sum_ = inv_sum.view(B, 1)
                has_other_color = (inv_sum_ - inv_b) > 0
                flip_extra = has_other_color & flip_enabled.view(B, 1)
                flip_extra &= (~place_enabled).view(B, 1)
                m_color = m_color | flip_extra

        # Qty mask: keep permissive (can be refined per-verb later)
        m_qty = th.ones((B, self.max_qtyp1), dtype=th.bool, device=device)

        # Safety: avoid all-false rows
        for head in (m_verb, m_node, m_color, m_qty):
            dead = (~head).all(dim=1)
            if dead.any():
                head[dead] = True

        return m_verb, m_node, m_color, m_qty

    # -------- forward --------
    def _masked_distribution(self, obs: th.Tensor, latent_pi: th.Tensor):
        logits_all = self.action_net(latent_pi)  # (B, sum(nvec))
        logits_heads = [logits_all[:, s:e] for (s, e) in self.slices]

        m_verb, m_node, m_color, m_qty = self._build_masks(obs)
        masks = [m_verb, m_node, m_color, m_qty]

        masked_heads = []
        for logit_head, mask_head in zip(logits_heads, masks):
            if mask_head.dtype != th.bool:
                mask_head = mask_head.bool()
            min_valid = logit_head.masked_fill(~mask_head, th.inf).amin(dim=1, keepdim=True)
            fill = (min_valid - 1.0).expand_as(logit_head)
            masked = th.where(mask_head, logit_head, fill)
            dead = (~mask_head).all(dim=1)
            if dead.any():
                masked[dead] = logit_head[dead]
            masked_heads.append(masked)

        masked_logits = th.cat(masked_heads, dim=1)
        return self.action_dist.proba_distribution(masked_logits)

    def _latent_pi_vf(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)
        return latent_pi, latent_vf

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        latent_pi, latent_vf = self._latent_pi_vf(obs)
        dist = self._masked_distribution(obs, latent_pi)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        latent_pi, latent_vf = self._latent_pi_vf(obs)
        dist = self._masked_distribution(obs, latent_pi)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    def get_distribution(self, obs: th.Tensor):
        latent_pi, _ = self._latent_pi_vf(obs)
        return self._masked_distribution(obs, latent_pi)
