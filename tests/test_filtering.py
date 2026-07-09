"""Tests for the shared min-length filtering helper."""

from udonpred.filtering import filter_rows_min_length, meets_min_length


def _row(length: int, id_: str = "x") -> dict:
    """A minimal dataset row: an ``x_0`` sequence of the given length."""
    return {"id": id_, "x_0": "A" * length}


class TestMeetsMinLength:
    def test_boundary_is_inclusive(self):
        # A 50-mer passes a threshold of exactly 50 (keep length >= N).
        assert meets_min_length(_row(50), 50) is True

    def test_shorter_is_dropped(self):
        assert meets_min_length(_row(49), 50) is False

    def test_longer_passes(self):
        assert meets_min_length(_row(200), 50) is True

    def test_zero_threshold_keeps_everything(self):
        assert meets_min_length(_row(1), 0) is True

    def test_none_threshold_keeps_everything(self):
        assert meets_min_length(_row(1), None) is True


class TestFilterRowsMinLength:
    def test_drops_short_keeps_long_with_boundary(self):
        rows = [_row(10, "a"), _row(50, "b"), _row(51, "c")]
        kept = filter_rows_min_length(rows, 50)
        assert [r["id"] for r in kept] == ["b", "c"]

    def test_zero_threshold_is_noop(self):
        rows = [_row(1), _row(2)]
        assert filter_rows_min_length(rows, 0) is rows

    def test_none_threshold_is_noop(self):
        rows = [_row(1), _row(2)]
        assert filter_rows_min_length(rows, None) is rows

    def test_all_shorter_yields_empty(self):
        rows = [_row(10), _row(20)]
        assert filter_rows_min_length(rows, 100) == []
