import math

import numpy as np
import pytest

from eurobot_world import (
    EurobotWorld,
    build_nodes,
    Col,
    PANTRY_CAP,
)
from rewards import DEFAULT_REWARD_CONFIG
from robot import Verb

REWARDS = DEFAULT_REWARD_CONFIG


class WaitPolicy:
    def next_action(self, actor_tag, world, state, rng):
        # minimal delay so the event loop advances deterministically
        return (int(Verb.WAIT), int(state.node), 0, 1)


def _set_wait_policy(world: EurobotWorld):
    world.yellow_policy = WaitPolicy()
    world.yellow.event = None


def _drain_events(world: EurobotWorld, max_steps: int = 10):
    for _ in range(max_steps):
        if world.blue.event is None and world.yellow.event is None:
            return
        dt, _ = world._advance_until_next()
        if dt <= 0:
            break


def test_world_initial_geometry(world):
    assert world.N == len(world.nodes)
    assert len(world.PANTRIES) == 10
    assert len(world.PICKUPS) == 8
    assert world.NEST_BLUE != world.NEST_YELL
    # distance matrix symmetric with zeros on diagonal
    assert np.allclose(world.D, world.D.T)
    assert np.allclose(np.diag(world.D), 0.0)


def test_reset_initial_state(world):
    world.reset(seed=42)
    assert world.t_left == pytest.approx(100.0)
    assert int(world.blue.node) == world.NEST_BLUE
    assert int(world.yellow.node) == world.NEST_YELL
    assert np.all(world.blue.inv == 0)
    assert np.all(world.yellow.inv == 0)
    assert np.all(world.pantries == 0)
    # each pickup starts with 2 blue + 2 yellow
    assert np.all(world.pickups[:, Col.BLUE] == 2)
    assert np.all(world.pickups[:, Col.YELLOW] == 2)
    assert np.all(world.pickups[:, Col.NEUTRAL] == 0)


def test_move_and_idle_penalty(world):
    _set_wait_policy(world)
    world.reset(seed=0)
    target = world.PICKUPS[0]
    r_move, done = world.step_blue((int(Verb.MOVE), target, 0, 0))
    _drain_events(world)
    assert not done
    assert int(world.blue.node) == target
    # time penalty is always negative
    assert r_move < 0

    # idle move -> immediate penalty
    idle_reward, done = world.step_blue((int(Verb.MOVE), target, 0, 0))
    assert idle_reward <= -REWARDS.invalid_action_penalty


def test_pick_success_and_empty_penalty(world):
    _set_wait_policy(world)
    world.reset(seed=1)
    pickup = world.PICKUPS[0]
    # move first
    world.step_blue((int(Verb.MOVE), pickup, int(Col.BLUE), 0))
    _drain_events(world)
    pre_stock = world.pickup_avail(pickup, Col.BLUE)
    reward, done = world.step_blue((int(Verb.PICK), pickup, int(Col.BLUE), 1))
    _drain_events(world)
    assert not done
    assert reward >= -1.0  # time penalty only
    assert world.blue.inv[Col.BLUE] == 1
    assert world.pickup_avail(pickup, Col.BLUE) == pre_stock - 1

    # picking a color with no stock should incur penalty
    penalty, _ = world.step_blue((int(Verb.PICK), pickup, int(Col.NEUTRAL), 1))
    _drain_events(world)
    extra = world.history[-1]["tag"] if world.history else ""
    assert penalty <= 0.0
    assert extra.endswith("pick_empty")
    assert world.blue.inv[Col.NEUTRAL] == 0


def test_place_to_pantry_and_nest(world):
    _set_wait_policy(world)
    world.reset(seed=2)
    pickup = world.PICKUPS[0]
    pantry = world.PANTRIES[0]
    nest = world.NEST_BLUE

    world.step_blue((int(Verb.MOVE), pickup, int(Col.BLUE), 0))
    _drain_events(world)
    world.step_blue((int(Verb.PICK), pickup, int(Col.BLUE), 2))
    _drain_events(world)
    assert world.blue.inv[Col.BLUE] == 2

    # place at pantry
    world.step_blue((int(Verb.MOVE), pantry, int(Col.BLUE), 0))
    _drain_events(world)
    reward_pan, _ = world.step_blue((int(Verb.PLACE), pantry, int(Col.BLUE), 1))
    _drain_events(world)
    assert reward_pan >= REWARDS.pantry_bonus - 1.0  # reward minus time penalty
    assert world.pantries[world.pantry_idx[pantry], Col.BLUE] == 1
    assert world.blue.inv[Col.BLUE] == 1

    # place remaining at nest
    world.step_blue((int(Verb.MOVE), nest, int(Col.BLUE), 0))
    _drain_events(world)
    reward_nest, _ = world.step_blue((int(Verb.PLACE), nest, int(Col.BLUE), 1))
    _drain_events(world)
    assert reward_nest >= REWARDS.nest_bonus - 1.0
    assert world.nest_blue_counted == 1
    assert world.blue.inv[Col.BLUE] == 0

    # placing with empty inventory → penalty
    penalty, _ = world.step_blue((int(Verb.PLACE), nest, int(Col.BLUE), 1))
    assert penalty <= -REWARDS.invalid_action_penalty


def test_flip_changes_inventory(world):
    _set_wait_policy(world)
    world.reset(seed=3)
    pickup = world.PICKUPS[0]
    world.step_blue((int(Verb.MOVE), pickup, int(Col.YELLOW), 0))
    _drain_events(world)
    world.step_blue((int(Verb.PICK), pickup, int(Col.YELLOW), 2))
    _drain_events(world)
    assert world.blue.inv[Col.YELLOW] == 2

    reward, _ = world.step_blue((int(Verb.FLIP), pickup, int(Col.BLUE), 2))
    assert reward >= -1.0
    assert world.blue.inv[Col.BLUE] == 2
    assert world.blue.inv[Col.YELLOW] == 0


def test_steal(world):
    _set_wait_policy(world)
    world.reset(seed=4)
    pantry = world.PANTRIES[0]
    idx = world.pantry_idx[pantry]
    world.pantries[idx, Col.YELLOW] = 3

    world.step_blue((int(Verb.MOVE), pantry, int(Col.YELLOW), 0))
    _drain_events(world)
    reward, _ = world.step_blue((int(Verb.STEAL), pantry, int(Col.YELLOW), 2))
    _drain_events(world)
    assert reward >= -1.0
    assert world.blue.inv[Col.YELLOW] == 2
    assert world.pantries[idx, Col.YELLOW] == 1

    # steal the last crate
    reward2, _ = world.step_blue((int(Verb.STEAL), pantry, int(Col.YELLOW), 2))
    _drain_events(world)
    assert world.blue.inv[Col.YELLOW] == 3
    assert world.pantries[idx, Col.YELLOW] == 0

    # stealing when empty incurs penalty
    penalty, _ = world.step_blue((int(Verb.STEAL), pantry, int(Col.YELLOW), 1))
    _drain_events(world)
    extra = world.history[-1]["tag"] if world.history else ""
    assert penalty <= 0.0
    assert extra.endswith("steal_empty")


def test_wait_penalty(world):
    _set_wait_policy(world)
    world.reset(seed=5)
    reward, _ = world.step_blue((int(Verb.WAIT), world.blue.node, 0, 1))
    assert reward <= -REWARDS.invalid_action_penalty


def test_terminal_bonus(world):
    world.reset(seed=6)
    idx = world.pantry_idx[world.PANTRIES[0]]
    world.pantries[idx, Col.BLUE] = 2
    world.pantries[idx, Col.YELLOW] = 1
    world.blue.node = world.NEST_BLUE
    bonus = world._terminal_bonus()
    assert bonus == REWARDS.interest_bonus + REWARDS.finish_in_nest_bonus

    blue_score, yellow_score = world.final_scores()
    assert blue_score == pytest.approx(
        REWARDS.pantry_bonus * 2
        + REWARDS.interest_bonus
        + REWARDS.finish_in_nest_bonus
    )
    # yellow robot stayed in its nest after reset
    assert yellow_score == pytest.approx(
        REWARDS.pantry_bonus * 1 + REWARDS.finish_in_nest_bonus
    )


def test_yellow_nest_counts(world):
    _set_wait_policy(world)
    world.reset(seed=8)
    world.yellow.inv[Col.YELLOW] = 1
    world._finish_event("yellow", int(Verb.PLACE), world.NEST_YELL, int(Col.YELLOW), 1, False, 0.0)
    assert world.nest_yellow_counted == 1
    _, yellow_score = world.final_scores()
    assert yellow_score >= REWARDS.nest_bonus


def test_helper_queries(world):
    world.reset(seed=7)
    # nearest pickup with stock should prefer local node when available
    start = world.PICKUPS[1]
    world.blue.node = start
    world.pickups[:] = 0
    world.pickups[world.pickup_idx[start], Col.BLUE] = 2
    world.pickups[0, Col.BLUE] = 5
    nearest = world.nearest_pickup_with_stock(start, Col.BLUE)
    assert nearest == start

    # best pantry should favor roomier option
    world.pantries[:] = 0
    spacious = world.PANTRIES[0]
    crowded = world.PANTRIES[1]
    world.pantries[world.pantry_idx[crowded], :] = np.array([PANTRY_CAP, 0, 0], dtype=world.pantries.dtype)
    world.blue.node = crowded
    best = world.best_pantry_for("blue", prefer_spread=True)
    assert best == spacious

    # best pantry to steal should pick stocked opponent pantry
    world.pantries[:] = 0
    world.allow_steal = True
    world.pantries[world.pantry_idx[spacious], Col.YELLOW] = 3
    steal_target = world.best_pantry_to_steal("blue", near_from=spacious)
    assert steal_target == spacious
