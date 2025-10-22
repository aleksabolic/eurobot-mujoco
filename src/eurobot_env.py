from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple
from eurobot_world import EurobotWorld, Col
from robot import Verb

class EurobotDiscreteEnv(gym.Env):
    """
    Single-agent (BLUE) Gym env wrapping EurobotWorld.
    Action = MultiDiscrete [verb(len(Verb)), node(N), color(2), qty(0..max_qty)].
    One env.step = one BLUE decision; 
    """
    metadata = {"render_modes": [], "name": "EurobotDiscrete-v1"}

    def __init__(self, seed: Optional[int]=None):
        super().__init__()
   
        blue_config_dir = "robot_configs/blue_robot.yaml"
        yellow_config_dir = "robot_configs/yellow_robot.yaml"

        self.world = EurobotWorld(blue_config_dir, yellow_config_dir, seed=seed)

        self.n_nodes = len(self.world.nodes)
        self.max_qty = int(self.world.blue.profile.max_action_qty)  # assumed same for obs space shape
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
        done = self.world.step_blue((v, n, c, q))
        return self._obs(), 0, done, False, {}

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
        o[self._sl_pan] = w.pantries.reshape(-1)
        o[self._sl_pick] = w.pickups.reshape(-1)
        o[self._sl_nest] = (float(w.nest_blue), float(w.nest_yellow))
        return o

    # --- planning utilities ---
    def get_state(self):
        """Serialize environment state for tree search cloning."""
        return dict(world=self.world.get_state())

    def set_state(self, state):
        """Restore environment state previously captured by :meth:`get_state`."""
        self.world.set_state(state["world"])
