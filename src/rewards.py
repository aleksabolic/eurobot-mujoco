'''
These rewards are taken from: 
https://www.eurobot.org/wp-content/uploads/2025/10/Eurobot2026_Rules_1.0_EN.pdf#page=17.07
'''
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class RewardConfig:
    nest_bonus: float = 2.0              # + per counted crate in nest (cap)
    pantry_bonus: float = 3.0            # + per valid pantry crate (BLUE placed by BLUE)
    interest_bonus: float = 5.0          # + per pantry where BLUE has strict BLUE majority at end
    time_penalty: float = 0.0           # - per second advanced
    invalid_action_penalty: float = 0.0 # discourages repeated invalid/empty actions
    finish_in_nest_bonus: float = 10.0   # + bonus when episode ends with BLUE inside its nest


DEFAULT_REWARD_CONFIG = RewardConfig()
