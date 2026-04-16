"""Shared pytest fixtures."""

from __future__ import annotations

import random

import pytest


@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(42)
