from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
from eurobot_world import EurobotWorld
from robot import RobotProfile
from policies import load_robot_config

class EurobotDiscreteEnv(gym.Env):
    """
    Single-agent (BLUE) Gym env wrapping EurobotWorld.
    Action = MultiDiscrete [verb(5), node(N), color(3), qty(0..max_qty)].
    One env.step = one BLUE decision; world advances to next event completion.
    """
    metadata = {"render_modes": [], "name": "EurobotDiscrete-v1"}

    def __init__(self, seed: Optional[int]=None):
        super().__init__()
        # TODO: move config loading out of ctor if needed
        blue_prof, blue_pol = load_robot_config("robot_configs/blue_robot.json")
        yellow_prof, yellow_pol = load_robot_config("robot_configs/yellow_robot.json")

        self.world = EurobotWorld(blue_prof, yellow_prof, seed=seed)
        self.world.yellow_policy = yellow_pol

        self.n_nodes = len(self.world.nodes)
        self.max_qty = int(blue_prof.max_action_qty)  # assumed same for obs space shape

        self.action_space = spaces.MultiDiscrete([5, self.n_nodes, 3, self.max_qty+1])
        # obs = [t_left(float scaled 0..100*10), blue_node, yellow_node,
        #        blue_inv(3), yellow_inv(3),
        #        pantries(#*3), pickups(#*3)]
        obs_dim = 3 + 3 + 3 + 3*len(self.world.PANTRIES) + 3*len(self.world.PICKUPS)
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self._obs_buf = np.zeros((obs_dim,), dtype=np.float32)
        # slices: t_by(3), blue_inv(3), yellow_inv(3), pantries, pickups
        i = 0
        self._sl_t_by = slice(i, i+3); i += 3
        self._sl_inv_b = slice(i, i+3); i += 3
        self._sl_inv_y = slice(i, i+3); i += 3
        self._sl_pan   = slice(i, i + 3*len(self.world.PANTRIES)); i += 3*len(self.world.PANTRIES)
        self._sl_pick  = slice(i, i + 3*len(self.world.PICKUPS));  i += 3*len(self.world.PICKUPS)

    def reset(self, seed: Optional[int]=None, options=None):
        self.world.reset(seed=seed)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        v, n, c, q = map(int, action)
        r, done = self.world.step_blue((v, n, c, q))
        info = {}
        done_flag = bool(done)
        if done_flag:
            blue_final, yellow_final = self.world.final_scores()
            info["blue_final_score"] = float(blue_final)
            info["yellow_final_score"] = float(yellow_final)
        return self._obs(), float(r), done_flag, False, info

    def _obs(self):
        w = self.world
        o = self._obs_buf
        # pack scalars
        o[self._sl_t_by] = (w.t_left * 10.0, float(w.blue.node), float(w.yellow.node))
        # inventories
        o[self._sl_inv_b] = w.blue.inv
        o[self._sl_inv_y] = w.yellow.inv
        # pantries/pickups flattened
        o[self._sl_pan]  = w.pantries.reshape(-1)
        o[self._sl_pick] = w.pickups.reshape(-1)
        return o
