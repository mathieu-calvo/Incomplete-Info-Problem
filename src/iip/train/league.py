"""League training — opponent pool with ELO-weighted sampling.

A League tracks a collection of named checkpoints. During training, the learner samples an
opponent with probability proportional to its ELO-softmax (toughest opponents preferred), which
mirrors AlphaStar's "main agents vs main exploiters" setup but much smaller.

Anyone can be promoted into the league after passing an eval gate (see `metrics/mbb.py` +
`scripts/evaluate_checkpoint.py`). The league state is just a JSON file so it round-trips
cleanly through CI artifacts and HF Hub.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LeagueEntry:
    name: str
    path: str
    elo: float = 1500.0
    hands_played: int = 0


@dataclass
class League:
    entries: list[LeagueEntry] = field(default_factory=list)

    def add(self, entry: LeagueEntry) -> None:
        self.entries.append(entry)

    def sample(self, rng: random.Random, temperature: float = 200.0) -> LeagueEntry:
        if not self.entries:
            raise RuntimeError("League is empty")
        weights = [math.exp(e.elo / temperature) for e in self.entries]
        total = sum(weights)
        r = rng.random() * total
        cum = 0.0
        for e, w in zip(self.entries, weights, strict=True):
            cum += w
            if r <= cum:
                return e
        return self.entries[-1]

    def update_elo(self, name_a: str, name_b: str, result_a: float, k: float = 16.0) -> None:
        a = self._get(name_a)
        b = self._get(name_b)
        expected_a = 1.0 / (1.0 + 10 ** ((b.elo - a.elo) / 400.0))
        a.elo += k * (result_a - expected_a)
        b.elo -= k * (result_a - expected_a)
        a.hands_played += 1
        b.hands_played += 1

    def _get(self, name: str) -> LeagueEntry:
        for e in self.entries:
            if e.name == name:
                return e
        raise KeyError(name)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps([e.__dict__ for e in self.entries], indent=2))

    @classmethod
    def load(cls, path: str | Path) -> League:
        data = json.loads(Path(path).read_text())
        return cls(entries=[LeagueEntry(**d) for d in data])
