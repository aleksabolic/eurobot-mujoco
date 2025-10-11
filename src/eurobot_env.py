from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
from eurobot_world import EurobotWorld, Col
from robot import RobotProfile, Verb
from policies import load_robot_config

class EurobotDiscreteEnv(gym.Env):
    """
    Single-agent (BLUE) Gym env wrapping EurobotWorld.
    Action = MultiDiscrete [verb(len(Verb)), node(N), color(2), qty(0..max_qty)].
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
        self.num_colors = len(Col)

        # TODO: remove + 1 in self.max_qty+1 as zero actions are invalid
        self.action_space = spaces.MultiDiscrete([len(Verb), self.n_nodes, self.num_colors, self.max_qty+1])
        # obs = [blue_node_one_hot(n_nodes),
        #        yellow_node_one_hot(n_nodes),
        #        blue_inv(num_colors),
        #        pantries(#*num_colors), pickups(#*num_colors)]
        obs_dim = (
            2 * self.n_nodes
            + self.num_colors
            + self.num_colors * len(self.world.PANTRIES)
            + self.num_colors * len(self.world.PICKUPS)
            + 2  # nest counts (blue, yellow)
        )
        #TODO: replace low, high to vectors, not scalars
        self.observation_space = spaces.Box(low=0.0, high=1e6, shape=(obs_dim,), dtype=np.float32)
        self._obs_buf = np.zeros((obs_dim,), dtype=np.float32)
        # slices: blue_node, yellow_node, blue_inv, pantries, pickups
        i = 0
        self._sl_node_blue = slice(i, i+self.n_nodes); i += self.n_nodes
        self._sl_node_yellow = slice(i, i+self.n_nodes); i += self.n_nodes
        self._sl_inv_b = slice(i, i+self.num_colors); i += self.num_colors
        self._sl_pan   = slice(i, i + self.num_colors*len(self.world.PANTRIES)); i += self.num_colors*len(self.world.PANTRIES)
        self._sl_pick  = slice(i, i + self.num_colors*len(self.world.PICKUPS));  i += self.num_colors*len(self.world.PICKUPS)
        self._sl_nest  = slice(i, i + 2); i += 2

    def reset(self, seed: Optional[int]=None, options=None):
        self.world.reset(seed=seed)
        return self._obs(), {}

    #TODO: remove final_score calculation on each step
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
        o.fill(0.0)
        # pack node positions as one-hot
        o[self._sl_node_blue.start + int(w.blue.node)] = 1.0
        o[self._sl_node_yellow.start + int(w.yellow.node)] = 1.0
        # inventories
        o[self._sl_inv_b] = w.blue.inv
        # pantries/pickups flattened
        o[self._sl_pan]  = w.pantries.reshape(-1)
        o[self._sl_pick] = w.pickups.reshape(-1)
        o[self._sl_nest] = (float(w.nest_blue_counted), float(w.nest_yellow_counted))
        return o
