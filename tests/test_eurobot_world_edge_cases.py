from dataclasses import replace

import pytest

from eurobot_world import Col, EurobotWorld
from policies import load_robot_config
from rewards import RewardConfig
from robot import Verb


class WaitPolicy:
    def next_action(self, actor_tag, world, state, rng):
        return (int(Verb.WAIT), int(state.node), int(Col.BLUE), 1)


def _drain_events(world: EurobotWorld, max_steps: int = 12) -> float:
    total = 0.0
    for _ in range(max_steps):
        if world.blue.event is None and world.yellow.event is None:
            return total
        dt, reward = world._advance_until_next()
        if dt <= 0.0:
            break
        total += reward
        total -= world.rewards.time_penalty * dt
    return total


def _make_world(*, blue_override=None, rewards=None, seed=123) -> EurobotWorld:
    blue_prof, blue_policy = load_robot_config("robot_configs/blue_robot.json")
    yellow_prof, yellow_policy = load_robot_config("robot_configs/yellow_robot.json")
    if blue_override:
        blue_prof = replace(blue_prof, **blue_override)
    world = EurobotWorld(
        blue_prof,
        yellow_prof,
        seed=seed,
        rewards=rewards if rewards is not None else RewardConfig(),
    )
    world.yellow_policy = WaitPolicy()
    world.reset(seed=seed)
    return world


def test_custom_reward_config_applied():
    cfg = RewardConfig(
        nest_bonus=5.0,
        pantry_bonus=9.0,
        interest_bonus=11.0,
        time_penalty=0.0,
        invalid_action_penalty=1.23,
        finish_in_nest_bonus=4.5,
    )
    world = _make_world(rewards=cfg)
    assert world.rewards is cfg

    pickup = world.PICKUPS[0]
    pantry = world.PANTRIES[0]
    nest = world.NEST_BLUE
    idx_pick = world.pickup_idx[pickup]
    world.pickups[idx_pick, Col.BLUE] = 5

    world.step_blue((int(Verb.MOVE), pickup, int(Col.BLUE), 0))
    _drain_events(world)
    world.step_blue((int(Verb.PICK), pickup, int(Col.BLUE), 2))
    _drain_events(world)

    world.step_blue((int(Verb.MOVE), pantry, int(Col.BLUE), 0))
    _drain_events(world)
    reward_place, _ = world.step_blue((int(Verb.PLACE), pantry, int(Col.BLUE), 1))
    reward_place += _drain_events(world)
    assert reward_place == pytest.approx(cfg.pantry_bonus)

    # place remaining crate into nest so inventory is empty
    world.step_blue((int(Verb.MOVE), nest, int(Col.BLUE), 0))
    _drain_events(world)
    reward_nest, _ = world.step_blue((int(Verb.PLACE), nest, int(Col.BLUE), 1))
    reward_nest += _drain_events(world)
    assert reward_nest == pytest.approx(cfg.nest_bonus)

    penalty, _ = world.step_blue((int(Verb.PLACE), pantry, int(Col.BLUE), 1))
    penalty += _drain_events(world)
    assert penalty == pytest.approx(-cfg.invalid_action_penalty)

    # ensure terminal bonus uses custom interest
    idx_pan = world.pantry_idx[pantry]
    world.pantries[idx_pan, Col.BLUE] = 2
    world.pantries[idx_pan, Col.YELLOW] = 1
    world.blue.node = nest
    assert world._terminal_bonus() == pytest.approx(cfg.interest_bonus + cfg.finish_in_nest_bonus)

    blue_score, yellow_score = world.final_scores()
    assert blue_score == pytest.approx(
        cfg.pantry_bonus * 2
        + cfg.nest_bonus * 1
        + cfg.interest_bonus
        + cfg.finish_in_nest_bonus
    )
    assert yellow_score == pytest.approx(
        cfg.pantry_bonus * 1 + cfg.finish_in_nest_bonus
    )


def test_pick_respects_capacity_and_max_qty():
    world = _make_world(blue_override=dict(capacity=3, max_action_qty=4))
    pickup = world.PICKUPS[0]
    idx_pick = world.pickup_idx[pickup]
    world.pickups[idx_pick, Col.BLUE] = 10

    world.step_blue((int(Verb.MOVE), pickup, int(Col.BLUE), 0))
    _drain_events(world)
    reward, _ = world.step_blue((int(Verb.PICK), pickup, int(Col.BLUE), 10))
    reward += _drain_events(world)
    # reward should only contain time penalty, ensure capacity respected
    assert world.blue.inv[Col.BLUE] == 3
    assert world.pickups[idx_pick, Col.BLUE] == 7
    assert reward <= 0.0

    follow_up, _ = world.step_blue((int(Verb.PICK), pickup, int(Col.BLUE), 1))
    follow_up += _drain_events(world)
    assert follow_up == pytest.approx(-world.rewards.invalid_action_penalty, abs=5e-3)
    assert world.blue.inv[Col.BLUE] == 3


def test_action_arguments_are_clamped():
    world = _make_world()
    last_node = world.N - 1
    first_pickup = world.PICKUPS[0]
    world.step_blue((int(Verb.MOVE), world.N + 5, 99, -4))
    _drain_events(world)
    assert int(world.blue.node) == last_node

    # move back to first pickup using negative index (should clamp to zero)
    world.step_blue((int(Verb.MOVE), -10, -3, 0))
    _drain_events(world)
    assert int(world.blue.node) == 0

    world.step_blue((int(Verb.MOVE), first_pickup, int(Col.BLUE), 0))
    _drain_events(world)
    idx_pick = world.pickup_idx[first_pickup]
    world.pickups[idx_pick, Col.YELLOW] = 3

    start_yellow = int(world.blue.inv[Col.YELLOW])
    world.step_blue((int(Verb.PICK), first_pickup, 99, 1))
    _drain_events(world)
    assert world.blue.inv[Col.YELLOW] == start_yellow + 1

    start_blue = int(world.blue.inv[Col.BLUE])
    world.step_blue((int(Verb.PICK), first_pickup, -5, 1))
    _drain_events(world)
    assert world.blue.inv[Col.BLUE] == start_blue + 1


def test_steal_blocked_when_disallowed():
    world = _make_world()
    world.allow_steal = False
    pantry = world.PANTRIES[0]
    idx = world.pantry_idx[pantry]
    world.pantries[idx, Col.YELLOW] = 3

    world.step_blue((int(Verb.MOVE), pantry, int(Col.YELLOW), 0))
    _drain_events(world)
    reward, _ = world.step_blue((int(Verb.STEAL), pantry, int(Col.YELLOW), 2))
    reward += _drain_events(world)
    assert reward == pytest.approx(-world.rewards.invalid_action_penalty, abs=5e-3)
    assert world.blue.inv[Col.YELLOW] == 0


def test_flip_blocked_when_robot_cannot_flip():
    world = _make_world(blue_override=dict(can_flip=False))
    pickup = world.PICKUPS[0]
    world.step_blue((int(Verb.MOVE), pickup, int(Col.YELLOW), 0))
    _drain_events(world)
    world.step_blue((int(Verb.PICK), pickup, int(Col.YELLOW), 2))
    _drain_events(world)

    reward, _ = world.step_blue((int(Verb.FLIP), pickup, int(Col.BLUE), 1))
    reward += _drain_events(world)
    assert reward == pytest.approx(-world.rewards.invalid_action_penalty, abs=5e-3)
    assert world.blue.inv[Col.YELLOW] == 2


def test_time_penalty_applies_per_elapsed_time():
    cfg = RewardConfig(time_penalty=0.5)
    world = _make_world(rewards=cfg)
    start = world.t_left
    target = world.PANTRIES[-1]
    reward, _ = world.step_blue((int(Verb.MOVE), target, int(Col.BLUE), 0))
    reward += _drain_events(world)
    elapsed = start - world.t_left
    assert elapsed > 0
    assert reward == pytest.approx(-cfg.time_penalty * elapsed)
