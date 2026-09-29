"""Experiment: ignore each protein's terminal labels in training (needs torch)."""

import pytest

torch = pytest.importorskip("torch")
trainer = pytest.importorskip("udonpred.training.model.trainer")

M = 999.0


def test_masks_both_termini_of_each_protein_within_its_own_length():
    # a padded batch: protein 0 has 7 residues, protein 1 has 4 (padded with 999)
    y = torch.tensor([
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3, 0.4, M, M, M],
    ])
    out = trainer.mask_termini(y, torch.tensor([7, 4]), 2)
    torch.testing.assert_close(out, torch.tensor([
        [M, M, 0.3, 0.4, 0.5, M, M],
        [M, M, M, M, M, M, M],
    ]))


def test_leaves_the_batch_untouched():
    y = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    trainer.mask_termini(y, torch.tensor([5]), 1)
    torch.testing.assert_close(y, torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]]))


def test_keeps_labels_that_were_already_missing():
    y = torch.tensor([[0.1, M, 0.3, 0.4, 0.5, 0.6]])
    out = trainer.mask_termini(y, torch.tensor([6]), 1)
    torch.testing.assert_close(out, torch.tensor([[M, M, 0.3, 0.4, 0.5, M]]))
