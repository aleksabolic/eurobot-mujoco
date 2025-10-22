from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Callable, Dict, Any
import numpy as np
from robot import RobotProfile, RobotState, Verb
from policies import GreedyStashPolicy, load_robot_config
from rewards import RewardConfig, DEFAULT_REWARD_CONFIG

# --------- Constraints ----------
NEST_CAP = 6              # max crates per nest
PANTRY_CAP = 8            # max crates per pantry (sum over colors)
TIME_LIMIT_S  = 100.0     # float seconds

# --------- Map ----------
class NodeType(IntEnum):
    NEST=0; PANTRY=1; PICKUP=2

@dataclass
class Node:
    name: str
    xy: np.ndarray
    kind: NodeType

PANTRIES_POS = {
    "A": np.array([-0.25,  0.45]), 
    "B": np.array([ 0.25,  0.45]),
    "C": np.array([-1.4, -0.20]), 
    "D": np.array([-0.70, -0.20]),
    "E": np.array([ 0.00, -0.20]), 
    "F": np.array([ 0.70, -0.20]),
    "G": np.array([ 1.4, -0.20]),
    "H": np.array([-0.80, -0.9]), 
    "I": np.array([ 0.00, -0.9]), 
    "J": np.array([ 0.80, -0.9]),
}
PICKUPS_POS = {
    "P1": np.array([-1.325,  0.20]), 
    "P2": np.array([ 1.325,  0.20]),
    "P3": np.array([-0.35, -0.20]), 
    "P4": np.array([ 0.35, -0.20]),
    "P5": np.array([-1.325, -0.60]), 
    "P6": np.array([ 1.325, -0.60]),
    "P7": np.array([-0.40, -0.825]), 
    "P8": np.array([ 0.40, -0.825]),
}
NESTS = {"blue": np.array([ 1.20,  0.775]),
         "yellow": np.array([-1.20,  0.775])}

def build_nodes() -> List[Node]:
    nodes: List[Node] = [
        Node("NestBlue",   NESTS["blue"].astype(np.float32),   NodeType.NEST),
        Node("NestYellow", NESTS["yellow"].astype(np.float32), NodeType.NEST),
    ]
    for k in ["A","B","C","D","E","F","G","H","I","J"]:
        nodes.append(Node(f"Pantry{k}", PANTRIES_POS[k].astype(np.float32), NodeType.PANTRY))
    for k in ["P1","P2","P3","P4","P5","P6","P7","P8"]:
        nodes.append(Node(k, PICKUPS_POS[k].astype(np.float32), NodeType.PICKUP))
    return nodes

# --------- Colors ----------
class Col(IntEnum):
    BLUE = 0
    YELLOW = 1

NODES = build_nodes()
PANTRIES = [i for i,n in enumerate(NODES) if n.kind==NodeType.PANTRY]
PICKUPS  = [i for i,n in enumerate(NODES) if n.kind==NodeType.PICKUP]
NEST_BLUE  = next(i for i,n in enumerate(NODES) if n.name=="NestBlue")
NEST_YELL  = next(i for i,n in enumerate(NODES) if n.name=="NestYellow")
XY = np.stack([n.xy for n in NODES], axis=0)
D = np.linalg.norm(XY[:,None,:] - XY[None,:,:], axis=-1).astype(np.float32)

NUM_COLORS = len(Col)

# --------- World ----------
class EurobotWorld:
    """
    Discrete, event-driven eurobot world.
    """
    def __init__(self,
                 blue_config_dir: str,
                 yellow_config_dir: str,
                 seed: Optional[int]=None,
                 rewards: RewardConfig = DEFAULT_REWARD_CONFIG,
                 allow_steal = True,
                 device: str = 'cpu'):
        
        self.rng = np.random.default_rng(seed)
        self.nodes = NODES
        self.N = len(NODES)
        self.D = D
        self.device = device

        # TODO: Move rewards to .json or .yaml
        self.rewards = rewards

        # TODO: Add default values
        blue_prof, blue_pol = load_robot_config(blue_config_dir) 
        yellow_prof, yellow_pol = load_robot_config(yellow_config_dir)

        self.blue = RobotState(
            tag="blue",
            profile=blue_prof,
            node=NEST_BLUE, 
            inv=np.zeros(NUM_COLORS, dtype=np.int8),
        )
        self.yellow = RobotState(
            tag="yellow",
            profile=yellow_prof,
            node=NEST_YELL, 
            inv=np.zeros(NUM_COLORS, dtype=np.int8),
        )

        # TODO: Move yellow policy to RobotState
        self.yellow_policy = yellow_pol

        self.allow_steal = allow_steal
        self.pantry_cap  = int(PANTRY_CAP)
        self.reset(seed=seed)

        # maps from node_id -> local index or -1 (O(1) lookup)
        self.pantry_idx = -np.ones(self.N, dtype=np.int32)
        self.pickup_idx = -np.ones(self.N, dtype=np.int32)
        for j, node_id in enumerate(PANTRIES):
            self.pantry_idx[node_id] = j
        for j, node_id in enumerate(PICKUPS):
            self.pickup_idx[node_id] = j

        self.PICKUPS = PICKUPS
        self.PANTRIES = PANTRIES
        self.NEST_BLUE = NEST_BLUE
        self.NEST_YELL = NEST_YELL

    # ----- lifecycle -----
    def reset(self, seed: Optional[int]=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t_left = float(TIME_LIMIT_S)

        self.pantries = np.zeros((len(PANTRIES), NUM_COLORS), dtype=np.int8)
        self.pickups  = np.zeros((len(PICKUPS),  NUM_COLORS), dtype=np.int8)
        self.nest_blue = 0
        self.nest_yellow = 0

        # initial pickup stock: 2 blue + 2 yellow each
        self.pickups[:, Col.BLUE]   = 2
        self.pickups[:, Col.YELLOW] = 2

        # episode history for renderer (list of shallow snapshots)
        self.history: List[Dict] = []
        self._snap("reset")

    # ----- public step for BLUE -----
    def step_blue(self, action):
        if self.blue.event is None:
            self._schedule(self.blue, action)

        if self.yellow.event is None:
            self._schedule_yellow_scripted()

        # Advance until BLUE's current event completes (decision epoch).
        while self.t_left > 1e-9 and self.blue.event is not None:
            self._advance_until_next()

            # Keep scripted YELLOW active while BLUE is still busy.
            if self.yellow.event is None and self.blue.event is not None:
                self._schedule_yellow_scripted()

        done = (self.t_left <= 1e-9)
        if done:
            self._snap("end")

        return bool(done)

    # ----- scheduling -----
    def _schedule(self, rob: RobotState, 
                  action: Tuple[int,int,int,int]) -> float:
        
        verb_i, node, color, qty = action

        verb = Verb(int(verb_i))
        node = int(node)
        color = int(color)
        qty = int(qty)

        if not self._check_valid_action(node, color, qty, rob):
            raise AssertionError(f"{verb_i, node, color, qty} is not a valid action")
            
        dist   = float(self.D[rob.node, node]) if node != rob.node else 0.0
        t_move = rob.profile.travel_time(dist) if verb in (Verb.PICK, Verb.PLACE, Verb.STEAL) and dist > 0 else 0.0
        t_hand = rob.profile.handle_time(verb, qty)
        t_total = t_move + t_hand

        # if "instant" work, just apply immediately without scheduling (TODO: remove?)
        if t_total <= 0.0:
            did_move = (t_move > 0.0)
            return self._finish_event(rob.tag, int(verb), node, color, qty, did_move)

        # else schedule as a compact tuple: [t_remaining, actor_id, verb, node, color, qty, did_move]
        actor_id = 0 if rob.tag=="blue" else 1
        did_move = (t_move > 0.0)
        rob.event = [t_total, actor_id, int(verb), node, color, qty, did_move]

    # ----- simple scripted yellow -----
    def _schedule_yellow_scripted(self):
        if self.yellow_policy is None:
            return
        a = self.yellow_policy.next_action("yellow", self, self.yellow, self.rng)
        if a is None:
            return
        self._schedule(self.yellow, a)

    # ----- advancing -----
    def _advance_until_next(self) -> Tuple[float, float]:
        tb = self.blue.event[0]   if self.blue.event   is not None else np.inf
        ty = self.yellow.event[0] if self.yellow.event is not None else np.inf
        dt = float(min(tb, ty))
        if not np.isfinite(dt) or dt <= 0.0:
            return 

        # Advance the match clock here ONLY
        self.t_left = max(0.0, self.t_left - dt)

        # Decrease remaining times (events are compact lists)
        if self.blue.event is not None:
            self.blue.event[0] -= dt
        if self.yellow.event is not None:
            self.yellow.event[0] -= dt

        eps = 1e-9
        if self.blue.event   is not None and self.blue.event[0] <= eps:
            # unpack and finish
            _, actor_id, verb, node, color, qty, did_move = self.blue.event
            self.blue.event = None
            self._finish_event("blue", verb, node, color, qty, bool(did_move))
        if self.yellow.event is not None and self.yellow.event[0] <= eps:
            _, actor_id, verb, node, color, qty, did_move = self.yellow.event
            self.yellow.event = None
            self._finish_event("yellow", verb, node, color, qty, bool(did_move))

    # ----- compact event executor -----
    def _finish_event(self, actor_tag: str, verb_i: int, node: int, color: int, qty: int, did_move: bool) -> float:
        rob   = self.blue   if actor_tag=="blue"   else self.yellow
        prof  = self.blue.profile if actor_tag=="blue" else self.yellow.profile

        if did_move:
            rob.node = node

        verb = Verb(verb_i)
        if verb == Verb.PICK:
            idx = self.pickup_idx[node]
            if idx == -1:
                self._snap(f"{actor_tag}_pick_invalid")
            can_take = int(self.pickups[idx, color])
            inv_sum = int(rob.inv[0]) + int(rob.inv[1])
            room = int(prof.capacity - inv_sum)
            take     = max(0, min(qty, can_take, room))
            if take > 0:
                self.pickups[idx, color] -= take
                rob.inv[color]           += take
                self._snap(f"{actor_tag}_pick")
            else:
                detail = dict(
                    reason="pickup_no_stock" if can_take <= 0 else "capacity_reached",
                    actor=actor_tag,
                    verb="PICK",
                    color=int(color),
                    qty=int(qty),
                    available=int(can_take),
                    capacity_left=int(room),
                    inventory=self._inventory_snapshot(rob.inv),
                )
                detail.update(self._describe_node(node))
                self._snap(f"{actor_tag}_pick_empty", invalid_detail=detail)

        if verb == Verb.PLACE:
            idx = self.pantry_idx[node]
            have = int(rob.inv[color])
            if have <= 0:
                detail = dict(
                    reason="empty_inventory",
                    actor=actor_tag,
                    verb="PLACE",
                    color=int(color),
                    qty=int(qty),
                    inventory=self._inventory_snapshot(rob.inv),
                )
                detail.update(self._describe_node(node))
                self._snap(f"{actor_tag}_place_empty", invalid_detail=detail)

            if idx != -1:  # placing at a pantry
                p0 = int(self.pantries[idx, 0]); p1 = int(self.pantries[idx, 1])
                total_here = p0 + p1
                room = max(0, self.pantry_cap - total_here)
                put = max(0, min(qty, have, room))
                if put > 0:
                    self.pantries[idx, color] += put
                    rob.inv[color]            -= put
                    self._snap(f"{actor_tag}_place")
                else:
                    detail = dict(
                        reason="no_room" if room <= 0 else "zero_qty",
                        actor=actor_tag,
                        verb="PLACE",
                        color=int(color),
                        qty=int(qty),
                        room=int(room),
                        inventory=self._inventory_snapshot(rob.inv),
                    )
                    detail.update(self._describe_node(node))
                    self._snap(f"{actor_tag}_place_full", invalid_detail=detail)

            if actor_tag=="blue" and node == NEST_BLUE:
                put = max(0, min(qty, have))
                #TODO: allow placing more than nest_cap but only reward up to nest_cap
                delta = min(put, max(0, NEST_CAP - int(self.nest_blue)))
                if delta > 0:
                    rob.inv[color] -= delta
                    self.nest_blue += delta
                    self._snap(f"{actor_tag}_place")
                else:
                    self._snap(f"{actor_tag}_place_nest_full")

            if actor_tag=="yellow" and node == NEST_YELL:
                put = max(0, min(qty, have))
                delta = min(put, max(0, NEST_CAP - int(self.nest_yellow)))
                if delta > 0:
                    rob.inv[color]   -= delta
                    self.nest_yellow += delta
                    self._snap(f"{actor_tag}_place")
                else:
                    self._snap(f"{actor_tag}_place_nest_full")

            if actor_tag == "blue":
                detail = dict(
                    reason="invalid_location",
                    actor=actor_tag,
                    verb="PLACE",
                    color=int(color),
                    qty=int(qty),
                    inventory=self._inventory_snapshot(rob.inv),
                )
                detail.update(self._describe_node(node))
                self._snap(f"{actor_tag}_place_invalid", invalid_detail=detail)
            else:
                self._snap(f"{actor_tag}_place_invalid")

        if verb == Verb.FLIP:
            if not prof.can_flip:
                self._snap(f"{actor_tag}_flip_blocked")
            # move from most abundant non-target to target
            src_candidates = [Col.BLUE, Col.YELLOW]
            if color in src_candidates: src_candidates.remove(Col(color))
            src = int(src_candidates[int(np.argmax(rob.inv[src_candidates]))])
            k = min(qty, int(rob.inv[src]))
            if k <= 0:
                detail = dict(
                    reason="flip_no_source",
                    actor=actor_tag,
                    verb="FLIP",
                    color=int(color),
                    qty=int(qty),
                    inventory=self._inventory_snapshot(rob.inv),
                )
                detail.update(self._describe_node(node))
                self._snap(f"{actor_tag}_flip_empty", invalid_detail=detail)
            rob.inv[src]   -= k
            rob.inv[color] += k
            self._snap(f"{actor_tag}_flip")

        if verb == Verb.STEAL:
            if not self.allow_steal:
                self._snap(f"{actor_tag}_steal_blocked")
            idx = self.pantry_idx[node]
            if idx == -1:
                self._snap(f"{actor_tag}_steal_invalid")
            have = int(self.pantries[idx, color])
            inv_sum = int(rob.inv[0]) + int(rob.inv[1])
            room = int(prof.capacity - inv_sum)
            take = max(0, min(qty, have, room))
            if take > 0:
                self.pantries[idx, color] -= take
                rob.inv[color]            += take
                self._snap(f"{actor_tag}_steal")
            else:
                detail = dict(
                    reason="steal_empty" if have <= 0 else "capacity_reached",
                    actor=actor_tag,
                    verb="STEAL",
                    color=int(color),
                    qty=int(qty),
                    available=int(have),
                    capacity_left=int(room),
                    inventory=self._inventory_snapshot(rob.inv),
                )
                detail.update(self._describe_node(node))
                self._snap(f"{actor_tag}_steal_empty", invalid_detail=detail)
    
    def _check_valid_action(self, node, color, qty, robot):
        if node < 0 or node > self.N - 1:
            return False
        
        if color < 0 or color > NUM_COLORS -1:
            return False
    
        if qty < 0 or qty > robot.profile.max_action_qty:
            return False
        
        return True
    
    def final_scores(self) -> Tuple[float, float]:
        r = self.rewards
        blue_score = self.pantries[:,Col.BLUE].sum() * r.pantry_bonus + self.nest_blue * r.nest_bonus 
        yellow_score = self.pantries[:,Col.YELLOW].sum() * r.pantry_bonus + self.nest_yellow * r.nest_bonus 

        blue_score   += r.interest_bonus * (self.pantries[:,Col.BLUE]>self.pantries[:,Col.YELLOW]).sum()
        yellow_score += r.interest_bonus * (self.pantries[:,Col.YELLOW]>self.pantries[:,Col.BLUE]).sum()

        if self.blue.node == NEST_BLUE:
            blue_score += r.finish_in_nest_bonus
        if self.yellow.node == NEST_YELL:
            yellow_score += r.finish_in_nest_bonus
        return float(blue_score), float(yellow_score)
    
    def _snap(self, tag: str, **extra):
        return # slows down learning
        blue_score, yellow_score = self.final_scores()
        snap = dict(
            tag=tag,
            t_left=float(self.t_left),
            blue_node=int(self.blue.node),
            yellow_node=int(self.yellow.node),
            blue_inv=self.blue.inv.copy(),
            yellow_inv=self.yellow.inv.copy(),
            pantries=self.pantries.copy(),
            pickups=self.pickups.copy(),
            nest_blue=int(self.nest_blue),
            nest_yellow=int(self.nest_yellow),
            blue_score=float(blue_score),
            yellow_score=float(yellow_score),
            blue_return=float(0),
        )
        if extra:
            snap["extra"] = extra
        self.history.append(snap)

    def _describe_node(self, node: int) -> Dict[str, Any]:
        node_idx = int(np.clip(node, 0, self.N - 1))
        n = self.nodes[node_idx]
        return dict(node=node_idx, node_name=str(n.name), node_kind=n.kind.name)

    def _inventory_snapshot(self, inv: np.ndarray) -> Dict[str, int]:
        return {Col(i).name.lower(): int(inv[i]) for i in range(NUM_COLORS)}

    # ----- serialization helpers -----
    def _robot_state_to_dict(self, robot: RobotState) -> Dict[str, Any]:
        return dict(
            tag=str(robot.tag),
            node=int(robot.node),
            inv=robot.inv.copy(),
            event=None if robot.event is None else tuple(robot.event),
        )

    def _robot_state_from_dict(self, robot: RobotState, data: Dict[str, Any]) -> None:
        robot.node = int(data["node"])
        robot.inv[...] = data["inv"]
        robot.event = None if data["event"] is None else list(data["event"])

    def get_state(self) -> Dict[str, Any]:
        """Return a deep copy of the mutable world state for planning algorithms."""
        return dict(
            t_left=float(self.t_left),
            pantries=self.pantries.copy(),
            pickups=self.pickups.copy(),
            nest_blue=int(self.nest_blue),
            nest_yellow=int(self.nest_yellow),
            blue=self._robot_state_to_dict(self.blue),
            yellow=self._robot_state_to_dict(self.yellow),
            allow_steal=bool(self.allow_steal),
            pantry_cap=int(self.pantry_cap),
            history=list(self.history),
            rng_state=self.rng.bit_generator.state,
        )

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore world state obtained from :meth:`get_state`."""
        self.t_left = float(state["t_left"])
        self.pantries[...] = state["pantries"]
        self.pickups[...] = state["pickups"]
        self.nest_blue = int(state["nest_blue"])
        self.nest_yellow = int(state["nest_yellow"])
        self._robot_state_from_dict(self.blue, state["blue"])
        self._robot_state_from_dict(self.yellow, state["yellow"])
        self.allow_steal = bool(state.get("allow_steal", self.allow_steal))
        self.pantry_cap = int(state.get("pantry_cap", self.pantry_cap))
        self.history = list(state.get("history", []))
        self.rng.bit_generator.state = state["rng_state"]
