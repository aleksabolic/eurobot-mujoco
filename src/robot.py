from __future__ import annotations
from dataclasses import dataclass, asdict
from enum import IntEnum
from typing import Callable, Optional, Tuple
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

class Verb(IntEnum):
    MOVE=0; PICK=1; PLACE=2; FLIP=3; WAIT=4; STEAL=5 

@dataclass
class RobotProfile:
    v_mean: float = 0.60
    t_startstop: float = 0.40
    t_pick_base: float = 0.10
    t_pick_per: float  = 0.15
    t_place_base: float = 0.10
    t_place_per: float  = 0.12
    t_flip_base: float  = 0.15
    t_flip_per: float   = 0.25
    can_flip: bool = True
    max_action_qty: int = 4
    capacity: int = 999
    v_max: float = 0.80   # m/s (cruise speed cap)
    a_max: float = 1.50   # m/s^2 (sym accel/decel)

    def travel_time(self, dist_m: float) -> float:
        """
        Symmetric trapezoidal profile (accel -> cruise -> decel).
        Falls back to triangular (no cruise) if the distance is too short to reach v_max.

        time = t_startstop + motion_time
        """
        d = max(0.0, float(dist_m))
        if d == 0.0:
            return self.t_startstop

        a = max(self.a_max, 1e-9)
        vmax = max(self.v_max, 1e-9)

        d_min = vmax * vmax / a 

        if d <= d_min + 1e-12:
            t_motion = 2.0 * np.sqrt(d / a)
        else:
            t_acc = vmax / a
            d_cruise = d - d_min
            t_cruise = d_cruise / vmax
            t_motion = 2.0 * t_acc + t_cruise

        return self.t_startstop + t_motion

    def handle_time(self, verb: Verb, qty: int) -> float:
        if verb == Verb.PICK:  return self.t_pick_base  + qty*self.t_pick_per
        if verb == Verb.PLACE: return self.t_place_base + qty*self.t_place_per
        if verb == Verb.FLIP:  return self.t_flip_base  + qty*self.t_flip_per
        if verb == Verb.STEAL: return self.t_pick_base  + qty*self.t_pick_per
        if verb == Verb.WAIT:  return 0.5 + 0.25*qty
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "RobotProfile":
        return RobotProfile(**d)
    
@dataclass
class RobotState:
    node: int
    inv: np.ndarray                      # shape (3,) -> [blue, yellow, neutral]
    event: Optional[Tuple[float, Callable[[], None]]] = None  # (remaining_time_s, on_finish)
