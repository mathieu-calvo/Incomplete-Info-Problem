"""Reservoir + rolling replay buffers.

Reservoir sampling lets Deep CFR treat the whole training history as one uniform sample,
which matters because both the advantage and strategy nets are regressed over *all* visited
infosets, not just the latest traversal. See Brown et al. 2019, Alg. 1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any


@dataclass
class ReservoirBuffer:
    capacity: int
    rng: random.Random = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()
        self._data: list[Any] = []
        self._seen: int = 0

    def add(self, item: Any) -> None:
        self._seen += 1
        if len(self._data) < self.capacity:
            self._data.append(item)
        else:
            i = self.rng.randint(0, self._seen - 1)
            if i < self.capacity:
                self._data[i] = item

    def sample(self, batch_size: int) -> list[Any]:
        if not self._data:
            return []
        k = min(batch_size, len(self._data))
        return self.rng.sample(self._data, k)

    def __len__(self) -> int:
        return len(self._data)
