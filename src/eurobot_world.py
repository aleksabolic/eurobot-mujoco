from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Callable, Dict
import numpy as np
from robot import RobotProfile, RobotState, Verb

# --------- Rules / scoring (tune here) ----------
P1_NEST       = 2         # + per counted crate in BLUE nest (cap)
NEST_CAP_BLUE = 6
P2_PANTRY     = 1         # + per valid pantry crate (BLUE or NEUTRAL placed by BLUE)
P3_INTEREST   = 3         # + per pantry where BLUE has strict BLUE majority at end
TIME_LIMIT_S  = 100.0     # float seconds
TIME_PENALTY  = 1e-3      # - per second advanced

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

# --------- Colors (no SIMA/rotten) ----------
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

        self.reset(seed=seed)

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

        self.nest_blue_counted = 0

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
        node = int(np.clip(node,  0, self.N-1))
        color = int(np.clip(color, 0, 2))
        qty   = int(np.clip(qty,   0, prof.max_action_qty))

        r = 0.0
        def finish_move():
            rob.node = node
            self._snap(f"{actor_tag}_move")

        def finish_pick():
            if self.nodes[rob.node].kind != NodeType.PICKUP: 
                self._snap(f"{actor_tag}_pick_invalid"); return
            idx = self.PICKUPS.index(rob.node)
            can_take = int(self.pickups[idx, color])
            room = int(prof.capacity - int(rob.inv.sum()))
            take = max(0, min(qty, can_take, room))
            if take > 0:
                self.pickups[idx, color] -= take
                rob.inv[color]           += take
                self._snap(f"{actor_tag}_pick")
            else:
                self._snap(f"{actor_tag}_pick_empty")


        def finish_place_blue():
            nd = self.nodes[rob.node]
            put = max(0, min(qty, int(rob.inv[color])))
            rob.inv[color] -= put
            if nd.kind == NodeType.PANTRY:
                k = self.PANTRIES.index(rob.node)
                self.pantries[k, color] += put
                if actor_tag=="blue" and color in (Col.BLUE, Col.NEUTRAL):
                    nonlocal r; r += P2_PANTRY * put
            elif actor_tag=="blue" and nd.kind==NodeType.NEST and rob.node==self.NEST_BLUE:
                delta = min(put, max(0, NEST_CAP_BLUE - int(self.nest_blue_counted)))
                self.nest_blue_counted += delta
                r += P1_NEST * delta
            self._snap(f"{actor_tag}_place")

        def finish_place_yellow():
            nd = self.nodes[rob.node]
            put = max(0, min(qty, int(rob.inv[color])))
            rob.inv[color] -= put
            if nd.kind == NodeType.PANTRY:
                k = self.PANTRIES.index(rob.node)
                self.pantries[k, color] += put
            self._snap(f"{actor_tag}_place")

        def finish_flip():
            if not prof.can_flip:
                self._snap(f"{actor_tag}_flip_blocked"); return
            # flip from the most abundant non-target pile to 'color'
            src_candidates = [Col.BLUE, Col.YELLOW, Col.NEUTRAL]
            if color in src_candidates: src_candidates.remove(Col(color))
            src = int(src_candidates[int(np.argmax(rob.inv[src_candidates]))])
            k = min(qty, int(rob.inv[src]))
            rob.inv[src]   -= k
            rob.inv[color] += k
            self._snap(f"{actor_tag}_flip")

        dist   = float(self.D[rob.node, node]) if node != rob.node else 0.0
        t_move = prof.travel_time(dist) if verb in (Verb.MOVE, Verb.PICK, Verb.PLACE) and dist > 0 else 0.0
        t_hand = prof.handle_time(verb, qty)
        t_total = t_move + t_hand

        if t_total <= 0.0:
            # MOVE to same node → no-op; other verbs qty=0 → execute immediately
            if verb == Verb.MOVE:
                return r
            def cb_immediate():
                if t_move > 0: finish_move()
                if verb == Verb.PICK:   finish_pick()
                elif verb == Verb.PLACE:
                    (finish_place_blue() if actor_tag=="blue" else finish_place_yellow())
                elif verb == Verb.FLIP: finish_flip()
                elif verb == Verb.WAIT: self._snap(f"{actor_tag}_wait")
            cb_immediate()
            return r

        def cb():
            if t_move > 0: finish_move()
            if verb == Verb.PICK:   finish_pick()
            elif verb == Verb.PLACE:
                (finish_place_blue() if actor_tag=="blue" else finish_place_yellow())
            elif verb == Verb.FLIP: finish_flip()
            elif verb == Verb.WAIT: self._snap(f"{actor_tag}_wait")

        rob.event = (t_total, cb)
        return r

    # ----- simple scripted yellow -----
    def _schedule_yellow_scripted(self):
        prof = self.yellow_prof
        inv = self.yellow.inv
        cap_left = int(prof.capacity - int(inv.sum()))

        # availability per color at each pickup
        availB = self.pickups[:, Col.BLUE]
        availY = self.pickups[:, Col.YELLOW]
        any_blue   = int(availB.sum())   > 0
        any_yellow = int(availY.sum())   > 0

        def nearest_with_stock(color: int) -> Optional[int]:
            pool = []
            if color == Col.BLUE:
                pool = [self.PICKUPS[i] for i in range(len(self.PICKUPS)) if self.pickups[i, Col.BLUE] > 0]
            else:
                pool = [self.PICKUPS[i] for i in range(len(self.PICKUPS)) if self.pickups[i, Col.YELLOW] > 0]
            if not pool: return None
            return self._nearest(self.yellow.node, pool)

        # 1) If carrying anything, bias to PLACE to pantries
        if inv.sum() > 0:
            node = self._best_pantry_for_yellow()
            qty  = min(prof.max_action_qty, int(inv[Col.YELLOW]))  # we only place yellow crates in this simple script
            if qty > 0:
                self._schedule(self.yellow, prof, (Verb.PLACE, node, int(Col.YELLOW), qty), "yellow")
                return
            # if somehow no yellow inv but has blue (could happen later), place blue instead
            qtyB = min(prof.max_action_qty, int(inv[Col.BLUE]))
            if qtyB > 0:
                self._schedule(self.yellow, prof, (Verb.PLACE, node, int(Col.BLUE), qtyB), "yellow")
                return
            # else just move toward pantry to avoid idling
            if node != self.yellow.node:
                self._schedule(self.yellow, prof, (Verb.MOVE, node, int(Col.YELLOW), 0), "yellow")
                return

        # 2) If inventory not full and there is stock somewhere, PICK from nearest stocked pickup
        if cap_left > 0 and (any_yellow or any_blue):
            # prefer yellow stock first; fall back to blue if no yellow left
            target_color = Col.YELLOW if any_yellow else Col.BLUE
            node = nearest_with_stock(target_color)
            if node is not None:
                # take up to available, but cap by action limit and capacity left
                idx = self.PICKUPS.index(node)
                available = int(self.pickups[idx, target_color])
                if available > 0:
                    qty = max(1, min(available, prof.max_action_qty, cap_left))
                    self._schedule(self.yellow, prof, (Verb.PICK, node, int(target_color), qty), "yellow")
                    return
            # if no node had stock for chosen color, try the other color explicitly
            other = Col.BLUE if target_color == Col.YELLOW else Col.YELLOW
            node2 = nearest_with_stock(other)
            if node2 is not None:
                idx = self.PICKUPS.index(node2)
                available = int(self.pickups[idx, other])
                qty = max(1, min(available, prof.max_action_qty, cap_left))
                self._schedule(self.yellow, prof, (Verb.PICK, node2, int(other), qty), "yellow")
                return

        # 3) No stock anywhere and inventory empty → wander gently (or wait a tick)
        # Move to the nearest pantry (visual change) or nearest pickup (explore)
        target = self._best_pantry_for_yellow()
        if target == self.yellow.node and self.PANTRIES:
            # go to a different pantry to avoid 0-time move
            alts = [p for p in self.PANTRIES if p != target]
            if alts:
                target = self._nearest(self.yellow.node, alts)
        if target != self.yellow.node:
            self._schedule(self.yellow, prof, (Verb.MOVE, target, int(Col.YELLOW), 0), "yellow")
        else:
            self._schedule(self.yellow, prof, (Verb.WAIT, self.yellow.node, int(Col.YELLOW), 1), "yellow")


    # ----- advancing -----
    def _advance_until_next(self) -> float:
        tb = self.blue.event[0]   if self.blue.event   is not None else np.inf
        ty = self.yellow.event[0] if self.yellow.event is not None else np.inf
        dt = float(min(tb, ty))
        if not np.isfinite(dt) or dt <= 0.0:
            return 0.0

        # Advance the match clock here ONLY
        self.t_left = max(0.0, self.t_left - dt)

        # Decrease remaining times
        if self.blue.event   is not None:
            self.blue   = RobotState(self.blue.node,   self.blue.inv,   (tb - dt, self.blue.event[1]))
        if self.yellow.event is not None:
            self.yellow = RobotState(self.yellow.node, self.yellow.inv, (ty - dt, self.yellow.event[1]))

        eps = 1e-9
        if self.blue.event   is not None and self.blue.event[0]   <= eps:
            cb = self.blue.event[1];   self.blue.event   = None; cb()
        if self.yellow.event is not None and self.yellow.event[0] <= eps:
            cb = self.yellow.event[1]; self.yellow.event = None; cb()
        return dt


    # ----- terminal bonus -----
    def _terminal_bonus(self) -> float:
        bonus = 0.0
        for k,_ in enumerate(self.PANTRIES):
            blue = int(self.pantries[k, Col.BLUE])
            yell = int(self.pantries[k, Col.YELLOW])
            if blue > yell: bonus += P3_INTEREST
        return bonus

    # ----- helpers -----
    def _snap(self, tag: str):
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
        if node not in self.PICKUPS: return 0
        return int(self.pickups[self.PICKUPS.index(node), color])

    def _best_pantry_for_yellow(self) -> int:
        scores=[]
        for k,i in enumerate(self.PANTRIES):
            y = int(self.pantries[k, Col.YELLOW]); b = int(self.pantries[k, Col.BLUE])
            margin = (y+1) - b
            dist = float(self.D[self.yellow.node, i])
            scores.append((1.5*margin - 0.5*dist, i))
        scores.sort(reverse=True, key=lambda t:t[0])
        return scores[0][1] if scores else self.PANTRIES[0]
