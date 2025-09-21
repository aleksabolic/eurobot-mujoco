import numpy as np

from policies import GreedyStashPolicy, ThiefPolicy, BalancedPolicy, load_robot_config
from robot import Verb
from eurobot_world import Col


class DummyRng:
    def __init__(self, value: float = 0.5):
        self._value = value

    def random(self):
        return self._value

    def integers(self, low, high):
        return low


def test_load_robot_config(robot_profiles):
    (blue_prof, blue_policy), (yellow_prof, yellow_policy) = robot_profiles
    assert blue_prof.max_action_qty > 0
    assert blue_policy is not None
    assert yellow_prof.capacity >= blue_prof.max_action_qty


def test_greedy_policy_moves_then_picks(world):
    policy = GreedyStashPolicy()
    action = policy.next_action("blue", world, world.blue, np.random.default_rng(0))
    assert action[0] in (Verb.MOVE, Verb.WAIT, Verb.PICK)
    # move toward a pickup when empty inventory
    if action[0] == Verb.MOVE:
        assert action[1] in world.PICKUPS

    # simulate arriving at target pickup
    target = world.PICKUPS[0]
    world.blue.node = target
    action_pick = policy.next_action("blue", world, world.blue, np.random.default_rng(1))
    assert action_pick[0] == Verb.PICK
    assert action_pick[1] == target


def test_thief_policy_prefers_steal(world):
    world.blue.node = world.PANTRIES[0]
    idx = world.pantry_idx[world.blue.node]
    world.pantries[idx, Col.YELLOW] = 3
    policy = ThiefPolicy()
    action = policy.next_action("blue", world, world.blue, np.random.default_rng(0))
    assert action[0] == Verb.STEAL
    assert action[1] == world.blue.node


def test_balanced_policy_flip_and_place(world):
    policy = BalancedPolicy()
    world.blue.inv[:] = np.array([2, 0, 0], dtype=world.blue.inv.dtype)
    # ensure flip branch triggers
    flip_rng = DummyRng(value=0.0)
    action_flip = policy.next_action("blue", world, world.blue, flip_rng)
    assert action_flip[0] == Verb.FLIP

    # ensure place branch when random >= 0.2
    world.blue.inv[:] = np.array([1, 0, 0], dtype=world.blue.inv.dtype)
    place_rng = DummyRng(value=0.5)
    action_place = policy.next_action("blue", world, world.blue, place_rng)
    assert action_place[0] == Verb.PLACE
    assert action_place[1] in world.PANTRIES
