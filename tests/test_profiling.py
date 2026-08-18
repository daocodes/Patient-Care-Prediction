"""Tests for the text profiler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from health_analytics.profiling import DatasetProfiler
from health_analytics.schema import ColumnClassifier


def _profile(frame: pd.DataFrame):
    classifier = ColumnClassifier()
    schema = classifier.classify(frame)
    return DatasetProfiler().profile(classifier.coerce(frame, schema), schema)


class TestBasics:
    def test_dimensions_are_recorded(self, mixed_frame: pd.DataFrame) -> None:
        profile = _profile(mixed_frame)
        assert profile.n_rows == len(mixed_frame)
        assert profile.n_columns == mixed_frame.shape[1]

    def test_missing_fractions_are_computed(self, mixed_frame: pd.DataFrame) -> None:
        profile = _profile(mixed_frame)
        assert profile.missing_fraction["systolic"] == 5 / len(mixed_frame)

    def test_complete_columns_are_omitted_from_the_missing_list(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        profile = _profile(mixed_frame)
        assert "PAT_ID" not in profile.columns_with_missing.index


class TestDominance:
    def test_a_dominated_column_is_detected(self, mixed_frame: pd.DataFrame) -> None:
        """`reminder` is 'email' in 58 of 60 rows."""
        profile = _profile(mixed_frame)
        assert "reminder" in profile.dominant_categories
        value, share = profile.dominant_categories["reminder"]
        assert value == "email"
        assert share > 0.9

    def test_a_balanced_column_is_not_flagged(self, mixed_frame: pd.DataFrame) -> None:
        profile = _profile(mixed_frame)
        assert "department" not in profile.dominant_categories

    def test_an_all_empty_column_does_not_raise(self) -> None:
        """The original indexed value_counts unconditionally and raised here."""
        frame = pd.DataFrame({"a": [1, 2, 3], "empty": [np.nan] * 3})
        profile = _profile(frame)
        assert "empty" not in profile.dominant_categories


class TestSkew:
    def test_a_skewed_column_is_detected(self) -> None:
        rng = np.random.default_rng(0)
        frame = pd.DataFrame(
            {"normal": rng.normal(size=500), "skewed": rng.exponential(size=500) ** 3}
        )
        profile = _profile(frame)
        assert "skewed" in profile.highly_skewed.index

    def test_skew_is_ordered_by_magnitude(self) -> None:
        rng = np.random.default_rng(1)
        frame = pd.DataFrame(
            {
                "a": rng.normal(size=400),
                "b": rng.exponential(size=400),
                "c": rng.exponential(size=400) ** 4,
            }
        )
        profile = _profile(frame)
        assert profile.skewness.index[0] == "c"


class TestRendering:
    def test_report_contains_every_section(self, mixed_frame: pd.DataFrame) -> None:
        text = _profile(mixed_frame).to_text()
        for heading in (
            "MISSING VALUES",
            "STRONGEST CORRELATIONS",
            "SKEWNESS",
            "CATEGORICAL DOMINANCE",
        ):
            assert heading in text

    def test_report_survives_a_frame_too_narrow_to_correlate(self) -> None:
        """Profiling must work on any input, including a single-column table."""
        text = _profile(pd.DataFrame({"only": [1.0, 2.0, 3.0]})).to_text()
        assert "Not enough numeric columns" in text
