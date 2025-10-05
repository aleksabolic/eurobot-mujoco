from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List, Sequence, Union
import json
import numpy as np
from robot import RobotProfile, Verb, RobotState

ActionLike = Union[Dict[str, Any], Sequence[Any]]

# -------- Policy interface --------
class Policy:
    name: str = "base"

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params or {}

    def next_action(self,
                    actor_tag: str,
                    world: "EurobotWorld", # type: ignore
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
        if kind == "static":       return StaticPolicy(params)
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
            other = 1 - target_color
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
            color = rng.integers(0, state.inv.shape[0])
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


class StaticPolicy(Policy):
    name = "static"

    # deterministic loop that alternates between two pickups/pantries
    DEFAULT_SEQUENCE: List[Dict[str, Any]] = [
        {"verb": "MOVE",  "node": "P1"},
        {"verb": "PICK",  "node": "P1",      "color": "yellow", "qty": 2},
        {"verb": "MOVE",  "node": "PantryA"},
        {"verb": "PLACE", "node": "PantryA",  "color": "yellow", "qty": 2},
        {"verb": "MOVE",  "node": "P3"},
        {"verb": "PICK",  "node": "P3",      "color": "yellow", "qty": 2},
        {"verb": "MOVE",  "node": "PantryB"},
        {"verb": "PLACE", "node": "PantryB",  "color": "yellow", "qty": 2},
    ]

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        seq_cfg = self.params.get("sequence")
        if seq_cfg is None:
            seq_cfg = self.DEFAULT_SEQUENCE
        if not isinstance(seq_cfg, (list, tuple)):
            raise ValueError("StaticPolicy requires 'sequence' to be a list or tuple of actions")
        self._sequence_cfg = list(seq_cfg)
        self._loop: bool = bool(self.params.get("loop", True))
        self._fallback_cfg: Optional[ActionLike] = self.params.get("fallback_action")

        self._compiled_plan: Optional[List[Tuple[int, int, int, int]]] = None
        self._node_lookup: Dict[str, int] = {}
        self._cursor: int = 0
        self._exhausted: bool = False
        self._last_t_left: Optional[float] = None

    def next_action(self,
                    actor_tag: str,
                    world: "EurobotWorld",  # type: ignore
                    state: RobotState,
                    rng: np.random.Generator) -> Tuple[int, int, int, int]:
        if self._compiled_plan is None:
            self._compile_plan(world)

        if self._last_t_left is None or world.t_left > self._last_t_left + 1e-6:
            self._cursor = 0
            self._exhausted = False
        self._last_t_left = float(world.t_left)

        if not self._compiled_plan:
            return self._fallback_action(world, state)

        if self._exhausted and not self._loop:
            return self._fallback_action(world, state)

        if self._cursor >= len(self._compiled_plan):
            if self._loop:
                self._cursor = 0
            else:
                self._exhausted = True
                return self._fallback_action(world, state)

        action = self._compiled_plan[self._cursor]
        self._cursor += 1
        return action

    # ---- helpers ----
    def _compile_plan(self, world: "EurobotWorld") -> None:  # type: ignore
        self._node_lookup = {n.name.lower(): idx for idx, n in enumerate(world.nodes)}
        self._compiled_plan = []
        for item in self._sequence_cfg:
            action = self._parse_action(world, item)
            self._compiled_plan.append(action)

    def _parse_action(self,
                      world: "EurobotWorld",  # type: ignore
                      spec: ActionLike,
                      *,
                      allow_missing_node: bool = False,
                      default_node: Optional[int] = None) -> Tuple[int, int, int, int]:
        if isinstance(spec, dict):
            verb_raw = spec.get("verb")
            node_raw = spec.get("node")
            color_raw = spec.get("color")
            qty_raw = spec.get("qty")
        elif isinstance(spec, (list, tuple)):
            if len(spec) != 4:
                raise ValueError("StaticPolicy sequence tuples must have four elements")
            verb_raw, node_raw, color_raw, qty_raw = spec
        else:
            raise TypeError("StaticPolicy sequence entries must be dicts or 4-tuples")

        verb = self._parse_verb(verb_raw)
        node = self._parse_node(world, node_raw, allow_missing=allow_missing_node, default_node=default_node)
        color = self._parse_color(color_raw, world)
        qty = self._parse_qty(qty_raw, verb, world)
        return (verb, node, color, qty)

    def _parse_verb(self, raw: Any) -> int:
        if isinstance(raw, str):
            key = raw.strip().upper()
            if key not in Verb.__members__:
                raise ValueError(f"Unknown verb '{raw}' for StaticPolicy")
            return int(Verb[key])
        try:
            return int(Verb(int(raw)))
        except (ValueError, TypeError):  # pragma: no cover - defensive
            raise ValueError(f"Invalid verb value '{raw}' for StaticPolicy")

    def _parse_node(self,
                    world: "EurobotWorld",
                    raw: Any,
                    *,
                    allow_missing: bool = False,
                    default_node: Optional[int] = None) -> int:
        if raw is None:
            if allow_missing and default_node is not None:
                return int(default_node)
            raise ValueError("StaticPolicy actions require a target node")

        if isinstance(raw, str):
            key = raw.strip().lower()
            if key not in self._node_lookup:
                raise ValueError(f"Unknown node name '{raw}' for StaticPolicy")
            return int(self._node_lookup[key])

        node = int(raw)
        if node < 0 or node >= int(world.N):
            raise ValueError(f"Node index {node} out of range for StaticPolicy")
        return node

    def _parse_color(self, raw: Any, world: "EurobotWorld") -> int:  # type: ignore
        if raw is None:
            return 0
        if isinstance(raw, str):
            key = raw.strip().lower()
            if key in ("blue", "b", "0"):
                return 0
            if key in ("yellow", "y", "1"):
                return 1
            raise ValueError(f"Unknown color '{raw}' for StaticPolicy")
        try:
            color = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid color value '{raw}' for StaticPolicy")
        if color < 0 or color >= world.blue.inv.shape[0]:  # num colors shared by both robots
            raise ValueError("Color indices must be non-negative for StaticPolicy")
        return color

    def _parse_qty(self, raw: Any, verb: int, world: "EurobotWorld") -> int:  # type: ignore
        max_qty = int(world.yellow_prof.max_action_qty)
        capacity = int(world.yellow_prof.capacity)
        verb_enum = Verb(int(verb))

        if verb_enum == Verb.MOVE:
            return 0

        if raw is None:
            if verb_enum in (Verb.PICK, Verb.PLACE, Verb.STEAL):
                raw = max_qty
            elif verb_enum == Verb.WAIT:
                raw = 1
            else:
                raw = 1

        try:
            qty = int(raw)
        except (ValueError, TypeError):
            raise ValueError(f"Invalid quantity '{raw}' for StaticPolicy")

        if qty < 0:
            raise ValueError("Quantities must be non-negative for StaticPolicy")

        if verb_enum in (Verb.PICK, Verb.PLACE, Verb.STEAL):
            qty = min(qty, max_qty, capacity)
        elif verb_enum == Verb.FLIP:
            qty = min(qty, max_qty)
        elif verb_enum == Verb.WAIT:
            qty = qty if qty > 0 else 1

        return qty

    def _fallback_action(self, world: "EurobotWorld", state: RobotState) -> Tuple[int, int, int, int]:  # type: ignore
        if self._fallback_cfg is None:
            return (int(Verb.WAIT), int(state.node), 0, 1)

        return self._parse_action(
            world,
            self._fallback_cfg,
            allow_missing_node=True,
            default_node=int(state.node),
        )

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
