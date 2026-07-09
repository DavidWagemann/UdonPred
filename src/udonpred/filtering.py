"""Min-length filtering of dataset rows, shared by training and eval.

A single rule -- keep proteins whose sequence (the row's ``x_0``) has at least
``min_length`` residues -- so the training config knob (``min_length`` in
``config/config.yaml``) and the eval CLI flag (``--min-length``) can't drift. A
falsy ``min_length`` (``0`` or ``None``) disables filtering.
"""


def meets_min_length(row, min_length) -> bool:
    """True if ``row``'s sequence (``x_0``) has at least ``min_length`` residues.

    A falsy ``min_length`` (``0``/``None``) disables filtering: every row passes.
    """
    if not min_length:
        return True
    return len(row["x_0"]) >= min_length


def filter_rows_min_length(rows, min_length):
    """Keep the ``rows`` whose ``x_0`` has at least ``min_length`` residues.

    Returns ``rows`` unchanged (same object) when ``min_length`` is falsy, so the
    common "no filter" path is a cheap no-op.
    """
    if not min_length:
        return rows
    return [row for row in rows if meets_min_length(row, min_length)]
