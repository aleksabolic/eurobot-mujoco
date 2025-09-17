from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import json
import numpy as np
from robot import RobotProfile, Verb, RobotState

# -------- Policy interface --------
class Policy:
    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def next_action(self,
                    actor_tag: str,
                    world: "EurobotWorld",
                    state,             # RobotState
                    rng: np.random.Generator) -> Tuple[int,int,int,int]:
        raise NotImplementedError

    def to_config(self) -> Dict[str, Any]:
        return dict(kind=self.name, params=self.params)

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Policy":
        kind = cfg.get("kind", "greedy_stash")
        params = cfg.get("params", {})
        if kind == "greedy_stash": return GreedyStashPolicy(params)
        if kind == "thief":        return ThiefPolicy(params)
        if kind == "balanced":     return BalancedPolicy(params)
        return GreedyStashPolicy(params)

# -------- Concrete policies --------
class GreedyStashPolicy(Policy):
    name = "greedy_stash"
    # Picks nearest stocked pickup (prefer YELLOW if actor is yellow), places at nearest pantry with room.
    def next_action(self, actor_tag, world, state, rng):
        prof = world.yellow_prof if actor_tag=="yellow" else world.blue_prof
        inv  = state.inv
        cap_left = int(prof.capacity - int(inv.sum()))

        # If carrying → place to best pantry with room
        if inv.sum() > 0:
            node = world.best_pantry_for(actor_tag, prefer_spread=False)
            qty  = min(prof.max_action_qty, int(inv[np.argmax(inv)]))
            return (int(Verb.PLACE), node, int(np.argmax(inv)), qty)

        # Else pick from nearest stocked pickup (color pref by actor)
        target_color = world.pref_pick_color(actor_tag)
        node = world.nearest_pickup_with_stock(state.node, target_color)
        if node is None:
            other = (1 - target_color) if target_color in (0,1) else 0
            node = world.nearest_pickup_with_stock(state.node, other)
            if node is None:
                # wander to pantry
                p = world.best_pantry_for(actor_tag, prefer_spread=True)
                if p != state.node:
                    return (int(Verb.MOVE), p, int(target_color), 0)
                return (int(Verb.WAIT), state.node, int(target_color), 1)
            target_color = other
        # move/pick
        if node == state.node:
            avail = world.pickup_avail(node, target_color)
            qty = max(1, min(avail, prof.max_action_qty, cap_left))
            return (int(Verb.PICK), node, int(target_color), qty)
        else:
            return (int(Verb.MOVE), node, int(target_color), 0)

class ThiefPolicy(Policy):
    name = "thief"
    # If nearby pantry has opponent majority and room → STEAL from it; else behave like greedy.
    def next_action(self, actor_tag, world, state, rng):
        prof = world.yellow_prof if actor_tag=="yellow" else world.blue_prof
        cap_left = int(prof.capacity - int(state.inv.sum()))

        # Try steal if allowed
        if world.allow_steal and cap_left > 0:
            target = world.best_pantry_to_steal(actor_tag, near_from=state.node)
            if target is not None:
                if target == state.node:
                    color = world.prefer_steal_color(actor_tag, target)
                    if world.pantry_avail(target, color) > 0:
                        qty = min(prof.max_action_qty, cap_left, world.pantry_avail(target, color))
                        return (int(Verb.STEAL), target, int(color), qty)
                else:
                    return (int(Verb.MOVE), target, 0, 0)

        # fallback
        return GreedyStashPolicy(self.params).next_action(actor_tag, world, state, rng)

class BalancedPolicy(Policy):
    name = "balanced"
    # Mix of stash, occasional flip, spread placements.
    def next_action(self, actor_tag, world, state, rng):
        prof = world.yellow_prof if actor_tag=="yellow" else world.blue_prof
        inv_sum = int(state.inv.sum())
        if inv_sum > 0 and rng.random() < 0.2 and prof.can_flip and inv_sum >= 2:
            # flip a few randomly
            color = rng.integers(0,3)
            qty = min(prof.max_action_qty, 2)
            return (int(Verb.FLIP), state.node, int(color), qty)
        # otherwise behave greedy but spread
        if inv_sum > 0:
            node = world.best_pantry_for(actor_tag, prefer_spread=True)
            color = int(np.argmax(state.inv))
            qty = min(prof.max_action_qty, int(state.inv[color]))
            return (int(Verb.PLACE), node, color, qty)
        # pick
        target_color = world.pref_pick_color(actor_tag)
        node = world.nearest_pickup_with_stock(state.node, target_color)
        if node is None:
            return (int(Verb.WAIT), state.node, 0, 1)
        if node == state.node:
            avail = world.pickup_avail(node, target_color)
            qty = max(1, min(avail, prof.max_action_qty, prof.capacity - inv_sum))
            return (int(Verb.PICK), node, int(target_color), qty)
        return (int(Verb.MOVE), node, int(target_color), 0)

# -------- Save / Load helpers --------
def save_robot_config(path: str,
                      profile: RobotProfile,
                      policy: Policy):
    cfg = {"profile": profile.to_dict(), "policy": policy.to_config()}
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)

def load_robot_config(path: str) -> Tuple[RobotProfile, Policy]:
    with open(path, "r") as f:
        cfg = json.load(f)
    profile = RobotProfile.from_dict(cfg["profile"])
    policy  = Policy.from_config(cfg["policy"])
    return profile, policy
