"""Tests for the imputation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from health_analytics.config import ImputationConfig
from health_analytics.imputation import ImputationPipeline, KNNNumericImputer
from health_analytics.schema import ColumnClassifier


@pytest.fixture
def pipeline() -> ImputationPipeline:
    return ImputationPipeline(ImputationConfig(n_neighbors=5))


class TestCompleteness:
    def test_no_gaps_remain(self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        assert not imputed.isna().to_numpy().any()

    def test_all_empty_column_gets_the_placeholder(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        assert (imputed["empty_col"] == "MISSING").all()


class TestStructurePreservation:
    def test_row_count_is_unchanged(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        assert len(imputed) == len(mixed_frame)

    def test_column_order_is_unchanged(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        assert list(imputed.columns) == list(mixed_frame.columns)

    def test_observed_values_are_never_overwritten(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        """Imputation may only fill gaps, never revise recorded measurements."""
        imputed, _ = pipeline.run(mixed_frame)
        observed = mixed_frame["diastolic"].notna()
        np.testing.assert_allclose(
            imputed.loc[observed, "diastolic"].astype(float),
            mixed_frame.loc[observed, "diastolic"].astype(float),
        )

    def test_source_frame_is_not_mutated(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        before = mixed_frame.copy()
        pipeline.run(mixed_frame)
        pd.testing.assert_frame_equal(mixed_frame, before)


class TestSkipColumns:
    def test_skipped_columns_keep_their_gaps(self, mixed_frame: pd.DataFrame) -> None:
        pipeline = ImputationPipeline(
            ImputationConfig(n_neighbors=5, skip_columns=("systolic",))
        )
        imputed, report = pipeline.run(mixed_frame)
        assert imputed["systolic"].isna().sum() == mixed_frame["systolic"].isna().sum()
        assert "systolic" not in report.filled_by_column


class TestIntegerPreservation:
    def test_integer_column_stays_integral(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        """A count of prior visits must not come back as 3.0000000004."""
        imputed, _ = pipeline.run(mixed_frame)
        values = imputed["visit_count"].astype(float).to_numpy()
        np.testing.assert_allclose(values, np.round(values))

    def test_continuous_column_stays_continuous(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        values = imputed["diastolic"].astype(float).to_numpy()
        assert not np.allclose(values, np.round(values))


class TestNeighborCapping:
    def test_k_is_capped_at_available_rows(self) -> None:
        """Requesting 20 neighbours from a 3-row frame must not raise."""
        tiny = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
        imputer = KNNNumericImputer(ImputationConfig(n_neighbors=20))
        result = imputer.fit_transform(tiny)
        assert imputer.neighbors_used == 2
        assert not result.isna().to_numpy().any()

    def test_invalid_neighbor_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_neighbors"):
            ImputationConfig(n_neighbors=0)


class TestReport:
    def test_report_counts_match_the_original_gaps(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        _, report = pipeline.run(mixed_frame)
        assert report.filled_by_column["systolic"] == 5
        assert report.filled_by_column["department"] == 3

    def test_total_matches_the_sum_of_the_parts(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        _, report = pipeline.run(mixed_frame)
        assert report.total_filled == sum(report.filled_by_column.values())

    def test_strategies_are_recorded(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        _, report = pipeline.run(mixed_frame)
        assert report.strategy_by_column["systolic"] == "knn"
        assert report.strategy_by_column["department"] == "mode"


class TestOriginalColumns:
    def test_backups_are_appended_when_requested(self, mixed_frame: pd.DataFrame) -> None:
        pipeline = ImputationPipeline(
            ImputationConfig(n_neighbors=5, keep_original_columns=True)
        )
        imputed, _ = pipeline.run(mixed_frame)
        assert "systolic_orig" in imputed.columns
        # The backup must retain the gaps, since that is the point of keeping it.
        assert imputed["systolic_orig"].isna().sum() == 5

    def test_backups_are_absent_by_default(
        self, pipeline: ImputationPipeline, mixed_frame: pd.DataFrame
    ) -> None:
        imputed, _ = pipeline.run(mixed_frame)
        assert not any(c.endswith("_orig") for c in imputed.columns)


class TestScaling:
    def test_large_scale_column_does_not_dominate_neighbours(self) -> None:
        """Standardisation is what keeps a big-unit column from owning the metric.

        ``cost`` spans tens of thousands while ``score`` spans single digits. Two
        rows are near-identical in cost; the gap in row 2's score must be filled
        from its cost-neighbour (row 0, score 1.0), not from the distant row 4.
        """
        frame = pd.DataFrame(
            {
                "cost": [10_000.0, 50_000.0, 10_100.0, 50_100.0, 90_000.0],
                "score": [1.0, 9.0, np.nan, 9.1, 5.0],
            }
        )
        result = KNNNumericImputer(ImputationConfig(n_neighbors=1)).fit_transform(frame)
        assert result.loc[2, "score"] == pytest.approx(1.0, abs=0.5)


class TestSchemaIntegration:
    def test_explicit_schema_is_respected(self, mixed_frame: pd.DataFrame) -> None:
        classifier = ColumnClassifier()
        schema = classifier.classify(mixed_frame)
        pipeline = ImputationPipeline(ImputationConfig(n_neighbors=5), classifier)
        imputed, _ = pipeline.run(mixed_frame, schema)
        assert not imputed.isna().to_numpy().any()
