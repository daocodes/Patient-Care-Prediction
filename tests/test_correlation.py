"""Tests for correlation analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from health_analytics.correlation import CorrelationAnalyzer
from health_analytics.schema import ColumnClassifier


@pytest.fixture
def analyzer() -> CorrelationAnalyzer:
    return CorrelationAnalyzer()


class TestRankPairs:
    def test_each_pair_appears_once(self) -> None:
        """A 4x4 matrix has 16 cells but only 6 distinct pairs."""
        matrix = pd.DataFrame(
            np.corrcoef(np.random.default_rng(0).normal(size=(4, 50))),
            index=list("abcd"),
            columns=list("abcd"),
        )
        pairs = CorrelationAnalyzer.rank_pairs(matrix)
        assert len(pairs) == 6

    def test_self_correlations_are_excluded(self) -> None:
        matrix = pd.DataFrame(np.eye(3), index=list("abc"), columns=list("abc"))
        pairs = CorrelationAnalyzer.rank_pairs(matrix)
        assert (pairs["feature_1"] != pairs["feature_2"]).all()

    def test_ordered_by_absolute_strength(self) -> None:
        """A strong negative correlation must outrank a weak positive one."""
        matrix = pd.DataFrame(
            [[1.0, -0.9, 0.1], [-0.9, 1.0, 0.2], [0.1, 0.2, 1.0]],
            index=list("abc"),
            columns=list("abc"),
        )
        pairs = CorrelationAnalyzer.rank_pairs(matrix)
        assert pairs.iloc[0]["correlation"] == pytest.approx(-0.9)
        assert pairs["abs_correlation"].is_monotonic_decreasing


class TestAnalyze:
    def test_finds_a_planted_relationship(self) -> None:
        rng = np.random.default_rng(1)
        base = rng.normal(size=200)
        frame = pd.DataFrame(
            {
                "a": base,
                "b": base * 2 + rng.normal(scale=0.01, size=200),  # near-perfect
                "c": rng.normal(size=200),  # unrelated
            }
        )
        classifier = ColumnClassifier()
        result = CorrelationAnalyzer().analyze(frame, classifier.classify(frame))

        top = result.ranked_pairs.iloc[0]
        assert {top["feature_1"], top["feature_2"]} == {"a", "b"}
        assert top["abs_correlation"] > 0.99

    def test_identifiers_are_excluded_by_default(self, mixed_frame: pd.DataFrame) -> None:
        classifier = ColumnClassifier()
        schema = classifier.classify(mixed_frame)
        result = CorrelationAnalyzer().analyze(classifier.coerce(mixed_frame, schema), schema)
        assert "PAT_ID" not in result.matrix.columns

    def test_constant_columns_are_dropped(self) -> None:
        """A constant column correlates with nothing and yields only NaN."""
        frame = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0], "flat": [7.0] * 4}
        )
        classifier = ColumnClassifier()
        result = CorrelationAnalyzer().analyze(frame, classifier.classify(frame))
        assert "flat" not in result.matrix.columns

    def test_too_few_numeric_columns_raises(self) -> None:
        frame = pd.DataFrame({"only": [1.0, 2.0, 3.0], "text": list("abc")})
        classifier = ColumnClassifier()
        with pytest.raises(ValueError, match="at least 2 numeric"):
            CorrelationAnalyzer().analyze(frame, classifier.classify(frame))

    def test_spearman_catches_a_monotone_nonlinear_pair(self) -> None:
        """The reason both methods are offered: Pearson understates a curve."""
        x = np.linspace(1, 10, 100)
        frame = pd.DataFrame({"x": x, "y": x**4})
        classifier = ColumnClassifier()
        schema = classifier.classify(frame)

        pearson = CorrelationAnalyzer().analyze(frame, schema, method="pearson")
        spearman = CorrelationAnalyzer().analyze(frame, schema, method="spearman")

        assert spearman.ranked_pairs.iloc[0]["abs_correlation"] == pytest.approx(1.0)
        assert pearson.ranked_pairs.iloc[0]["abs_correlation"] < 1.0


class TestResultHelpers:
    @pytest.fixture
    def result(self):
        """Three features: a-b nearly collinear, c independent."""
        rng = np.random.default_rng(2)
        base = rng.normal(size=300)
        frame = pd.DataFrame(
            {
                "a": base,
                "b": base + rng.normal(scale=0.05, size=300),
                "c": rng.normal(size=300),
            }
        )
        classifier = ColumnClassifier()
        return CorrelationAnalyzer().analyze(frame, classifier.classify(frame))

    def test_above_returns_only_pairs_meeting_the_threshold(self, result) -> None:
        strong = result.above(0.9)
        assert len(strong) == 1
        assert {strong.iloc[0]["feature_1"], strong.iloc[0]["feature_2"]} == {"a", "b"}

    def test_above_can_return_nothing(self, result) -> None:
        assert result.above(0.999).empty

    def test_top_limits_the_row_count(self, result) -> None:
        assert len(result.top(2)) == 2

    def test_method_is_recorded_on_the_result(self, result) -> None:
        assert result.method == "pearson"
