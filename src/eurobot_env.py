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

        # TODO: Pomeri ovo u train.py ovako je malo seljacki
        blue_prof, blue_pol = load_robot_config("robot_configs/blue_robot.json") 
        yellow_prof, yellow_pol = load_robot_config("robot_configs/yellow_robot.json")

        self.world = EurobotWorld(blue_prof, yellow_prof, seed=seed)
        self.world.yellow_policy = yellow_pol

        self.n_nodes = len(self.world.nodes)
        self.max_qty = blue_prof.max_action_qty  # assumed same for obs space shape

        self.action_space = spaces.MultiDiscrete([5, self.n_nodes, 3, self.max_qty+1])
        # obs = [t_left(float scaled 0..100*10), blue_node, yellow_node,
        #        blue_inv(3), yellow_inv(3),
        #        pantries(#*3), pickups(#*3)]
        obs_dim = 3 + 3 + 3 + 3*len(self.world.PANTRIES) + 3*len(self.world.PICKUPS)
        self.observation_space = spaces.Box(low=0, high=65535, shape=(obs_dim,), dtype=np.uint16)

    def reset(self, seed: Optional[int]=None, options=None):
        self.world.reset(seed=seed)
        return self._obs(), {}

    def step(self, action: np.ndarray):
        v, n, c, q = map(int, action)
        r, done = self.world.step_blue((v, n, c, q))
        return self._obs(), float(r), bool(done), False, {}

    def _obs(self):
        # scale t_left (float) into an int for compact obs (×10 precision)
        t10 = int(round(self.world.t_left * 10.0))
        vec = [t10, self.world.blue.node, self.world.yellow.node]
        vec += list(self.world.blue.inv)
        vec += list(self.world.yellow.inv) # TODO: Remove this, agent shouldn't know inv of the opponent
        for i in range(len(self.world.PANTRIES)): vec += list(self.world.pantries[i])
        for i in range(len(self.world.PICKUPS)):  vec += list(self.world.pickups[i])
        return np.array(vec, dtype=np.uint16)
