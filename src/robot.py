from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Optional, Tuple
import numpy as np

class Verb(IntEnum):
    MOVE=0; PICK=1; PLACE=2; FLIP=3; WAIT=4

@dataclass
class RobotProfile:
    # Kinematics and handling times
    v_mean: float = 0.60        # m/s
    t_startstop: float = 0.40   # s (per move)
    t_pick_base: float = 0.10
    t_pick_per: float  = 0.15
    t_place_base: float = 0.10
    t_place_per: float  = 0.12
    t_flip_base: float  = 0.15
    t_flip_per: float   = 0.25
    can_flip: bool = True
    max_action_qty: int = 4     # per Pick/Place/Flip op
    capacity: int = 999         # inventory cap; set small if you want to model trays

    def travel_time(self, dist_m: float) -> float:
        return self.t_startstop + dist_m / max(self.v_mean, 1e-6)

    def handle_time(self, verb: Verb, qty: int) -> float:
        if verb == Verb.PICK:  return self.t_pick_base  + qty*self.t_pick_per
        if verb == Verb.PLACE: return self.t_place_base + qty*self.t_place_per
        if verb == Verb.FLIP:  return self.t_flip_base  + qty*self.t_flip_per
        if verb == Verb.WAIT:  return 0.5 + 0.25*qty
        return 0.0

@dataclass
class RobotState:
    node: int
    inv: np.ndarray                      # shape (3,) -> [blue, yellow, neutral]
    event: Optional[Tuple[float, Callable[[], None]]] = None  # (remaining_time_s, on_finish)
