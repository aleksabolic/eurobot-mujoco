from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np


@dataclass
class ReplaySample:
    observations: np.ndarray
    policies: np.ndarray
    values: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self._obs: Deque[np.ndarray] = deque(maxlen=self.capacity)
        self._pi: Deque[np.ndarray] = deque(maxlen=self.capacity)
        self._values: Deque[float] = deque(maxlen=self.capacity)

    def add(self, obs: np.ndarray, policy: np.ndarray, value: float) -> None:
        self._obs.append(np.array(obs, copy=True))
        self._pi.append(np.array(policy, copy=True))
        self._values.append(float(value))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._obs)

    def sample(self, batch_size: int) -> ReplaySample:
        batch_size = min(int(batch_size), len(self._obs))
        if batch_size <= 0:
            raise RuntimeError("ReplayBuffer.sample called with an empty buffer.")
        idxs = random.sample(range(len(self._obs)), batch_size)
        obs = np.stack([self._obs[i] for i in idxs], axis=0)
        pi = np.stack([self._pi[i] for i in idxs], axis=0)
        values = np.asarray([self._values[i] for i in idxs], dtype=np.float32)
        return ReplaySample(obs, pi, values)
