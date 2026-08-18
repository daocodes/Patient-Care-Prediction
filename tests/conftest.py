"""Shared fixtures.

Every fixture builds synthetic data. No test reads the real dataset -- the
tests must be runnable by anyone who clones the repository, and the real data
is confidential and never present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator, so a failing test fails the same way twice."""
    return np.random.default_rng(0)


@pytest.fixture
def mixed_frame(rng: np.random.Generator) -> pd.DataFrame:
    """A frame shaped like the real data: IDs, outcomes, mixed features, gaps.

    Deliberately awkward in the ways the real export is: a numeric column
    holding a few text entries, a column that is entirely empty, and a
    categorical column dominated by one value.
    """
    n = 60
    frame = pd.DataFrame(
        {
            "PAT_ID": range(1, n + 1),
            "Y1": rng.normal(0.5, 0.2, n),
            "Y2": rng.integers(0, 2, n),
            "systolic": rng.normal(120, 15, n),
            "diastolic": rng.normal(80, 10, n),
            "visit_count": rng.integers(0, 9, n).astype(float),
            "department": rng.choice(["GEN", "SURG", "PSY"], n),
            "reminder": ["email"] * (n - 2) + ["text", "phone"],
            "empty_col": [np.nan] * n,
        }
    )

    # Punch holes in a few columns, at fixed positions so assertions can
    # target specific cells.
    frame.loc[0:4, "systolic"] = np.nan
    frame.loc[10:12, "department"] = np.nan
    frame.loc[20:22, "visit_count"] = np.nan
    return frame


@pytest.fixture
def text_frame() -> pd.DataFrame:
    """A frame as it arrives from a raw read: every cell an object.

    Includes the placeholder null tokens ("", "nan", "NA") that a text read
    preserves as literal strings.
    """
    return pd.DataFrame(
        {
            "mostly_numeric": ["1", "2", "3", "4", "unknown", "6", "7", "8", "9", "10"],
            "clean_numeric": [str(i) for i in range(10)],
            "categorical": list("aabbccddee"),
            "with_null_tokens": ["1", "", "3", "nan", "5", "NA", "7", "8", "9", "10"],
        },
        dtype=object,
    )
