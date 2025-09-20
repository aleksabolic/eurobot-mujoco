from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Callable, Dict
import numpy as np
from robot import RobotProfile, RobotState, Verb
from policies import GreedyStashPolicy

# --------- Rules / scoring (tune here) ----------
P1_NEST       = 2         # + per counted crate in BLUE nest (cap)
NEST_CAP_BLUE = 6
P2_PANTRY     = 1         # + per valid pantry crate (BLUE or NEUTRAL placed by BLUE)
P3_INTEREST   = 3         # + per pantry where BLUE has strict BLUE majority at end
TIME_LIMIT_S  = 100.0     # float seconds
TIME_PENALTY  = 1e-3      # - per second advanced
PANTRY_CAP = 8            # max crates per pantry (sum over colors)
ALLOW_STEAL = True  

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
    "P1": np.array([-1.375,  0.20]), 
    "P2": np.array([ 1.375,  0.20]),
    "P3": np.array([-0.35, -0.20]), 
    "P4": np.array([ 0.35, -0.20]),
    "P5": np.array([-1.375, -0.60]), 
    "P6": np.array([ 1.375, -0.60]),
    "P7": np.array([-0.40, -0.875]), 
    "P8": np.array([ 0.40, -0.875]),
}
NESTS = {"blue": np.array([ 1.20,  0.80]),
         "yellow": np.array([-1.20,  0.80])}

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
    BLUE=0; YELLOW=1; NEUTRAL=2

# --------- World ----------
class EurobotWorld:
    """
    Discrete, event-driven eurobot world.
    - Two robots with independent RobotProfile (BLUE learns, YELLOW scripted by default)
    - Quantity+color pick/place, optional flip, per-robot timings.
    """
    def __init__(self,
                 blue_profile: RobotProfile,
                 yellow_profile: RobotProfile,
                 seed: Optional[int]=None):
        self.rng = np.random.default_rng(seed)
        self.nodes = build_nodes()
        self.N = len(self.nodes)

        XY = np.stack([n.xy for n in self.nodes], axis=0)
        self.D = np.linalg.norm(XY[:,None,:] - XY[None,:,:], axis=-1).astype(np.float32)

        self.PANTRIES = [i for i,n in enumerate(self.nodes) if n.kind==NodeType.PANTRY]
        self.PICKUPS  = [i for i,n in enumerate(self.nodes) if n.kind==NodeType.PICKUP]
        self.NEST_BLUE  = next(i for i,n in enumerate(self.nodes) if n.name=="NestBlue")
        self.NEST_YELL  = next(i for i,n in enumerate(self.nodes) if n.name=="NestYellow")

        self.blue_prof   = blue_profile
        self.yellow_prof = yellow_profile

        # TODO: move this in __init__ args
        self.yellow_policy = GreedyStashPolicy()

        self.allow_steal = ALLOW_STEAL
        self.pantry_cap  = int(PANTRY_CAP)
        self.reset(seed=seed)

        # maps from node_id -> local index or -1 (O(1) lookup)
        self.pantry_idx = -np.ones(self.N, dtype=np.int32)
        self.pickup_idx = -np.ones(self.N, dtype=np.int32)
        for j, node_id in enumerate(self.PANTRIES):
            self.pantry_idx[node_id] = j
        for j, node_id in enumerate(self.PICKUPS):
            self.pickup_idx[node_id] = j

    # ----- lifecycle -----
    def reset(self, seed: Optional[int]=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        self.t_left = float(TIME_LIMIT_S)

        self.blue   = RobotState(node=self.NEST_BLUE, inv=np.zeros(3, np.int16))
        self.yellow = RobotState(node=self.NEST_YELL, inv=np.zeros(3, np.int16))

        self.pantries = np.zeros((len(self.PANTRIES), 3), dtype=np.int16)
        self.pickups  = np.zeros((len(self.PICKUPS),  3), dtype=np.int16)
        # initial pickup stock: 2 blue + 2 yellow each
        self.pickups[:, Col.BLUE]   = 2
        self.pickups[:, Col.YELLOW] = 2

        self.nest_blue_counted = 0 # ?

        # episode history for renderer (list of shallow snapshots)
        self.history: List[Dict] = []
        self._snap("reset")

    # ----- public step for BLUE -----
    def step_blue(self, action):
        r = 0.0
        if self.blue.event is None:
            r += self._schedule(self.blue, self.blue_prof, action, actor_tag="blue")
        if self.yellow.event is None:
            self._schedule_yellow_scripted()

        dt = self._advance_until_next()  # dt > 0 when something happens

        r -= TIME_PENALTY * dt

        done = (self.t_left <= 1e-9)
        if done:
            r += self._terminal_bonus()
            self._snap("end")
        return float(r), bool(done)


    # ----- scheduling -----
    def _schedule(self, rob: RobotState, prof: RobotProfile,
                  action: Tuple[int,int,int,int], actor_tag: str) -> float:
        verb_i, node, color, qty = action
        verb = Verb(int(verb_i))
        node = int(node)
        if node < 0:
            node = 0
        elif node > self.N - 1:
            node = self.N - 1
        color = int(color)
        if color < 0:
            color = 0
        elif color > 2:
            color = 2
        qty = int(qty)
        if qty < 0:
            qty = 0
        elif qty > prof.max_action_qty:
            qty = prof.max_action_qty

        r = 0.0
        dist   = float(self.D[rob.node, node]) if node != rob.node else 0.0
        t_move = prof.travel_time(dist) if verb in (Verb.MOVE, Verb.PICK, Verb.PLACE) and dist > 0 else 0.0
        t_hand = prof.handle_time(verb, qty)
        t_total = t_move + t_hand

        # if "instant" work, just apply immediately without scheduling
        if t_total <= 0.0:
            # MOVE to same node → no-op; other verbs qty=0 → execute immediately
            if verb == Verb.MOVE:
                return r
            did_move = (t_move > 0.0)
            return self._finish_event(actor_tag, int(verb), node, color, qty, did_move, r)

        # else schedule as a compact tuple: [t_remaining, actor_id, verb, node, color, qty, did_move]
        actor_id = 0 if actor_tag=="blue" else 1
        did_move = (t_move > 0.0)
        rob.event = [t_total, actor_id, int(verb), node, color, qty, did_move]
        return r

    # ----- simple scripted yellow -----
    def _schedule_yellow_scripted(self):
        from policies import Policy  # type: ignore
        a = self.yellow_policy.next_action("yellow", self, self.yellow, self.rng)
        self._schedule(self.yellow, self.yellow_prof, a, "yellow")

    # ----- advancing -----
    def _advance_until_next(self) -> float:
        tb = self.blue.event[0]   if self.blue.event   is not None else np.inf
        ty = self.yellow.event[0] if self.yellow.event is not None else np.inf
        dt = float(min(tb, ty))
        if not np.isfinite(dt) or dt <= 0.0:
            return 0.0

        # Advance the match clock here ONLY
        self.t_left = max(0.0, self.t_left - dt)

        # Decrease remaining times (events are compact lists)
        if self.blue.event is not None:
            self.blue.event[0] -= dt
        if self.yellow.event is not None:
            self.yellow.event[0] -= dt

        eps = 1e-9
        if self.blue.event   is not None and self.blue.event[0]   <= eps:
            # unpack and finish
            _, actor_id, verb, node, color, qty, did_move = self.blue.event
            self.blue.event = None
            _ = self._finish_event("blue", verb, node, color, qty, bool(did_move), 0.0)
        if self.yellow.event is not None and self.yellow.event[0] <= eps:
            _, actor_id, verb, node, color, qty, did_move = self.yellow.event
            self.yellow.event = None
            _ = self._finish_event("yellow", verb, node, color, qty, bool(did_move), 0.0)
        return dt

    # ----- compact event executor -----
    def _finish_event(self, actor_tag: str, verb_i: int, node: int, color: int, qty: int, did_move: bool, r_in: float) -> float:
        r = r_in
        rob   = self.blue   if actor_tag=="blue"   else self.yellow
        prof  = self.blue_prof if actor_tag=="blue" else self.yellow_prof

        if did_move:
            rob.node = node
            self._snap(f"{actor_tag}_move")

        verb = Verb(verb_i)
        if verb == Verb.PICK:
            idx = self.pickup_idx[node]
            if idx == -1:
                self._snap(f"{actor_tag}_pick_invalid"); return r
            can_take = int(self.pickups[idx, color])
            # rob.inv is length-3: avoid tiny numpy sum allocation by summing scalars
            inv0 = int(rob.inv[0]); inv1 = int(rob.inv[1]); inv2 = int(rob.inv[2])
            inv_sum = inv0 + inv1 + inv2
            room = int(prof.capacity - inv_sum)
            take     = max(0, min(qty, can_take, room))
            if take > 0:
                self.pickups[idx, color] -= take
                rob.inv[color]           += take
                self._snap(f"{actor_tag}_pick")
            else:
                self._snap(f"{actor_tag}_pick_empty")
            return r

        if verb == Verb.PLACE:
            idx = self.pantry_idx[node]
            have = int(rob.inv[color])
            if have <= 0:
                self._snap(f"{actor_tag}_place_empty")
                return r

            if idx != -1:  # placing at a pantry
                p0 = int(self.pantries[idx, 0]); p1 = int(self.pantries[idx, 1]); p2 = int(self.pantries[idx, 2])
                total_here = p0 + p1 + p2
                room = max(0, self.pantry_cap - total_here)
                put = max(0, min(qty, have, room))
                if put > 0:
                    self.pantries[idx, color] += put
                    rob.inv[color]            -= put
                    if actor_tag=="blue" and color in (Col.BLUE, Col.NEUTRAL):
                        r += P2_PANTRY * put
                    self._snap(f"{actor_tag}_place")
                else:
                    self._snap(f"{actor_tag}_place_full")
                return r

            if actor_tag=="blue" and node == self.NEST_BLUE:
                put = max(0, min(qty, have))
                delta = min(put, max(0, NEST_CAP_BLUE - int(self.nest_blue_counted)))
                if delta > 0:
                    rob.inv[color]         -= delta
                    self.nest_blue_counted += delta
                    r += P1_NEST * delta
                    self._snap(f"{actor_tag}_place")
                else:
                    self._snap(f"{actor_tag}_place_nest_full")
                return r

            self._snap(f"{actor_tag}_place_invalid")
            return r

        if verb == Verb.FLIP:
            if not prof.can_flip:
                self._snap(f"{actor_tag}_flip_blocked"); return r
            # move from most abundant non-target to target
            src_candidates = [Col.BLUE, Col.YELLOW, Col.NEUTRAL]
            if color in src_candidates: src_candidates.remove(Col(color))
            src = int(src_candidates[int(np.argmax(rob.inv[src_candidates]))])
            k = min(qty, int(rob.inv[src]))
            rob.inv[src]   -= k
            rob.inv[color] += k
            self._snap(f"{actor_tag}_flip")
            return r

        if verb == Verb.STEAL:
            if not self.allow_steal:
                self._snap(f"{actor_tag}_steal_blocked"); return r
            idx = self.pantry_idx[node]
            if idx == -1:
                self._snap(f"{actor_tag}_steal_invalid"); return r
            have = int(self.pantries[idx, color])
            inv0 = int(rob.inv[0]); inv1 = int(rob.inv[1]); inv2 = int(rob.inv[2])
            inv_sum = inv0 + inv1 + inv2
            room = int(prof.capacity - inv_sum)
            take = max(0, min(qty, have, room))
            if take > 0:
                self.pantries[idx, color] -= take
                rob.inv[color]            += take
                self._snap(f"{actor_tag}_steal")
            else:
                self._snap(f"{actor_tag}_steal_empty")
            return r

        if verb == Verb.WAIT:
            self._snap(f"{actor_tag}_wait")
            return r

        return r

    # ----- terminal bonus -----
    def _terminal_bonus(self) -> float:
        bonus = 0.0
        for k,_ in enumerate(self.PANTRIES):
            blue = int(self.pantries[k, Col.BLUE])
            yell = int(self.pantries[k, Col.YELLOW])
            if blue > yell: bonus += P3_INTEREST
        if int(self.blue.node) == self.NEST_BLUE:
            bonus += P1_NEST
        return bonus

    #TODO move this somewhere else
    # ----- helpers -----
    def _snap(self, tag: str):
        # return # slows down learning 
        self.history.append(dict(
            tag=tag,
            t_left=float(self.t_left),
            blue_node=int(self.blue.node),
            yellow_node=int(self.yellow.node),
            blue_inv=self.blue.inv.copy(),
            yellow_inv=self.yellow.inv.copy(),
            pantries=self.pantries.copy(),
            pickups=self.pickups.copy(),
        ))

    def _nearest(self, start: int, pool: List[int]) -> int:
        i = int(np.argmin(self.D[start, pool] + 1e-6))
        return int(pool[i])

    def _pickup_avail(self, node: int, color: int) -> int:
        idx = self.pickup_idx[node]
        return 0 if idx == -1 else int(self.pickups[idx, color])
    
    # --- policy helpers (thin wrappers) ---
    def pickup_avail(self, node: int, color: int) -> int:
        return self._pickup_avail(node, color)

    def pantry_avail(self, node: int, color: int) -> int:
        idx = self.pantry_idx[node]
        return 0 if idx == -1 else int(self.pantries[idx, color])

    def nearest_pickup_with_stock(self, start: int, color: int) -> Optional[int]:
        pool = [self.PICKUPS[i] for i in range(len(self.PICKUPS)) if self.pickups[i, color] > 0]
        if not pool:
            return None
        return self._nearest(start, pool)

    def pref_pick_color(self, actor_tag: str) -> int:
        # yellow prefers YELLOW, blue prefers BLUE, fall back to BLUE
        return int(Col.YELLOW if actor_tag=="yellow" else Col.BLUE)

    def best_pantry_for(self, actor_tag: str, prefer_spread: bool=False) -> int:
        # simple score: distance penalty + room left; prefer spread penalizes current majority
        scores=[]
        rob = (self.yellow if actor_tag=="yellow" else self.blue)
        for k,i in enumerate(self.PANTRIES):
            # pantries[k] is length-3; sum explicitly
            p0 = int(self.pantries[k, 0]); p1 = int(self.pantries[k, 1]); p2 = int(self.pantries[k, 2])
            total = p0 + p1 + p2
            room  = max(0, self.pantry_cap - total) if self.pantry_cap > 0 else 999
            if room <= 0:
                continue
            dist = float(self.D[rob.node, i])
            spread_pen = 0.0
            if prefer_spread:
                # penalize heavy current color presence
                spread_pen = float(self.pantries[k, Col.BLUE]**2 + self.pantries[k, Col.YELLOW]**2 + self.pantries[k, Col.NEUTRAL]**2) * 0.05
            scores.append((1.0*room - 0.5*dist - spread_pen, i))
        if not scores:
            return self.PANTRIES[0]
        scores.sort(reverse=True, key=lambda t: t[0])
        return scores[0][1]

    def best_pantry_to_steal(self, actor_tag: str, near_from: int) -> Optional[int]:
        if not self.allow_steal: return None
        opp_color = Col.BLUE if actor_tag=="yellow" else Col.YELLOW
        cand=[]
        for k,i in enumerate(self.PANTRIES):
            if int(self.pantries[k, opp_color]) > 0:
                dist = float(self.D[near_from, i])
                margin = int(self.pantries[k, opp_color]) - int(self.pantries[k, 1-opp_color])
                cand.append((1.5*margin - 0.4*dist, i))
        if not cand: return None
        cand.sort(reverse=True, key=lambda t: t[0])
        return cand[0][1]

    def prefer_steal_color(self, actor_tag: str, pantry_node: int) -> int:
        # steal opponent color first; if empty, steal neutral
        opp = Col.BLUE if actor_tag=="yellow" else Col.YELLOW
        k = self.pantry_idx[pantry_node]
        if k != -1 and int(self.pantries[k, opp]) > 0:
            return int(opp)
        if k != -1 and int(self.pantries[k, Col.NEUTRAL]) > 0:
            return int(Col.NEUTRAL)
        return int(opp)
