"""Unit tests for :mod:`udonpred.training.run`.

Importing the training entry point pulls in the heavy ``udonpred[training]``
stack (transformers, wandb, ...), so the module is skipped when that stack is
absent. Only the pure batch-size arithmetic is exercised here.
"""

import pytest

run = pytest.importorskip("udonpred.training.run")


def test_batch_size_fits_in_single_batch():
    # total <= max -> no gradient accumulation
    assert run.calculate_batch_sizes(4, 8) == (4, 1)


def test_batch_size_splits_into_accumulation_steps():
    # total > max -> (max_single, total // max_single)
    assert run.calculate_batch_sizes(16, 4) == (4, 4)


def test_batch_size_exact_multiple():
    assert run.calculate_batch_sizes(8, 8) == (8, 1)
