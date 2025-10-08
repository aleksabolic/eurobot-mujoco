import math

import pytest

from robot import RobotProfile, Verb


@pytest.fixture()
def profile():
    return RobotProfile()


def test_travel_time_zero_distance(profile):
    assert profile.travel_time(0.0) == pytest.approx(profile.t_startstop)


def test_travel_time_triangular_motion(profile):
    d = 0.1  # shorter than distance needed to reach cruise speed
    a = profile.a_max
    expected_motion = 2.0 * math.sqrt(d / a)
    expected_total = profile.t_startstop + expected_motion
    assert profile.travel_time(d) == pytest.approx(expected_total, rel=1e-5)


def test_travel_time_trapezoidal_motion(profile):
    d = 1.0  # large enough to include cruise phase
    vmax = profile.v_max
    a = profile.a_max
    d_min = vmax * vmax / a
    t_acc = vmax / a
    t_cruise = (d - d_min) / vmax
    expected_motion = 2.0 * t_acc + t_cruise
    expected_total = profile.t_startstop + expected_motion
    assert profile.travel_time(d) == pytest.approx(expected_total, rel=1e-5)


def test_handle_time_mappings(profile):
    assert profile.handle_time(Verb.PICK, 3) == pytest.approx(
        profile.t_pick_base + 3 * profile.t_pick_per
    )
    assert profile.handle_time(Verb.PLACE, 2) == pytest.approx(
        profile.t_place_base + 2 * profile.t_place_per
    )
    assert profile.handle_time(Verb.FLIP, 1) == pytest.approx(
        profile.t_flip_base + profile.t_flip_per
    )
    assert profile.handle_time(Verb.STEAL, 4) == pytest.approx(
        profile.t_pick_base + 4 * profile.t_pick_per
    )


def test_handle_time_zero_quantity(profile):
    with pytest.raises(AssertionError):
        profile.handle_time(Verb.PICK, 0)
    with pytest.raises(AssertionError):
        profile.handle_time(Verb.PLACE, 0)
    with pytest.raises(AssertionError):
        profile.handle_time(Verb.STEAL, 0)
    with pytest.raises(AssertionError):
        profile.handle_time(Verb.FLIP, 0)
