"""The shared CLI contract, verified without importing torch.

The on-the-fly runner pulls torch in at import time, so its own parser can only
be exercised where the ``embedding`` extra is installed
(see ``tests/test_embedding_backbone.py``). These tests cover the shared parent
parser itself, which is what keeps the two CLIs from drifting apart.
"""

import argparse

import pytest

from udonpred.caid.predict import build_parser as caid_parser
from udonpred.cli import common_parser

SHARED_FLAGS = {
    "--revision",
    "--target",
    "--output",
    "--device",
    "--threads",
    "--smooth",
    "--normalize",
}


def _flags(parser):
    return {
        opt
        for action in parser._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }


def test_common_parser_defines_every_shared_flag():
    parser = common_parser(device_choices=["cpu", "cuda"], device_default="cpu")
    assert SHARED_FLAGS <= _flags(parser)


def test_caid_parser_inherits_the_shared_flags():
    # the CAID runner adds only --embeddings on top of the shared contract
    assert SHARED_FLAGS <= _flags(caid_parser())
    assert "--embeddings" in _flags(caid_parser())


def test_shared_defaults():
    parser = common_parser(device_choices=["cpu", "cuda"], device_default="cpu")
    args = parser.parse_args(["in.fasta"])
    assert args.target == ["trizod"]
    assert args.smooth == 1.5
    assert args.normalize is True
    assert args.device == "cpu"
    assert args.output is None
    assert args.model_dir is None
    assert args.threads is None


def test_device_choices_are_per_runner():
    caid = common_parser(device_choices=["cpu", "cuda"], device_default="cpu")
    on_the_fly = common_parser(
        device_choices=["auto", "cpu", "cuda"], device_default="auto"
    )
    assert on_the_fly.parse_args(["in.fasta"]).device == "auto"
    assert on_the_fly.parse_args(["in.fasta", "-d", "auto"]).device == "auto"
    with pytest.raises(SystemExit):
        caid.parse_args(["in.fasta", "-d", "auto"])


def test_normalize_is_a_boolean_optional_flag():
    parser = common_parser(device_choices=["cpu"], device_default="cpu")
    assert parser.parse_args(["in.fasta", "--no-normalize"]).normalize is False
    assert parser.parse_args(["in.fasta", "--normalize"]).normalize is True


def test_target_accepts_multiple_values_and_all():
    parser = common_parser(device_choices=["cpu"], device_default="cpu")
    args = parser.parse_args(["in.fasta", "-t", "trizod", "chezod"])
    assert args.target == ["trizod", "chezod"]
    assert parser.parse_args(["in.fasta", "-t", "all"]).target == ["all"]


def test_shared_parser_is_reusable_as_a_parent():
    # add_help=False, so it can be composed without clashing on -h
    parser = common_parser(device_choices=["cpu"], device_default="cpu")
    composed = argparse.ArgumentParser(parents=[parser])
    assert "--target" in _flags(composed)
