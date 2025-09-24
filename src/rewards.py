from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RewardConfig:
    nest_bonus: float = 1.0              # + per counted crate in nest (cap)
    pantry_bonus: float = 2.0            # + per valid pantry crate (BLUE or NEUTRAL placed by BLUE)
    interest_bonus: float = 3.0          # + per pantry where BLUE has strict BLUE majority at end
    time_penalty: float = 1e-3           # - per second advanced
    invalid_action_penalty: float = 0.30 # discourages repeated invalid/empty actions


DEFAULT_REWARD_CONFIG = RewardConfig()

