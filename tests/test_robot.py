import numpy as np

from robot import RobotProfile, Verb


def test_travel_time_and_handle_time():
    profile = RobotProfile()
    assert profile.travel_time(0.0) == profile.t_startstop
    long_distance = 5.0
    assert profile.travel_time(long_distance) > profile.t_startstop

    assert profile.handle_time(Verb.PICK, 2) == profile.t_pick_base + 2 * profile.t_pick_per
    assert profile.handle_time(Verb.PLACE, 3) == profile.t_place_base + 3 * profile.t_place_per
    assert profile.handle_time(Verb.WAIT, 4) == 0.5 + 0.25 * 4


def test_profile_serialization():
    profile = RobotProfile(v_mean=0.7, can_flip=False)
    d = profile.to_dict()
    restored = RobotProfile.from_dict(d)
    assert restored == profile
