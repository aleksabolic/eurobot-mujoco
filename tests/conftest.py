import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from policies import load_robot_config
from eurobot_world import EurobotWorld
from eurobot_env import EurobotDiscreteEnv


@pytest.fixture()
def robot_profiles():
    blue_prof, blue_policy = load_robot_config("robot_configs/blue_robot.yaml")
    yellow_prof, yellow_policy = load_robot_config("robot_configs/yellow_robot.yaml")
    return (blue_prof, blue_policy), (yellow_prof, yellow_policy)


@pytest.fixture()
def world(robot_profiles):
    (blue_prof, _), (yellow_prof, yellow_policy) = robot_profiles
    world = EurobotWorld(blue_prof, yellow_prof, seed=123)
    world.yellow_policy = yellow_policy
    world.reset(seed=123)
    return world


@pytest.fixture()
def env():
    env = EurobotDiscreteEnv(seed=123)
    obs, _ = env.reset(seed=123)
    return env


@pytest.fixture()
def mask_cfg(world, robot_profiles):
    (blue_prof, _), _ = robot_profiles
    return dict(
        n_pantries=len(world.PANTRIES),
        n_pickups=len(world.PICKUPS),
        n_nodes=world.N,
        max_qty=int(blue_prof.max_action_qty),
        capacity=int(blue_prof.capacity),
        pantry_cap=int(world.pantry_cap),
        allow_steal=bool(world.allow_steal),
        can_flip=bool(blue_prof.can_flip),
        pantry_idx=world.pantry_idx.tolist(),
        pickup_idx=world.pickup_idx.tolist(),
        pantry_nodes=world.PANTRIES,
        pickup_nodes=world.PICKUPS,
        nest_blue=world.NEST_BLUE,
        nest_yellow=world.NEST_YELL,
    )
