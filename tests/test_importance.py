"""Tests for task inference, encoding, and the importance study."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from health_analytics.config import ImportanceConfig
from health_analytics.importance import (
    CategoricalEncoder,
    FeatureImportanceStudy,
    GainImportance,
    TaskInference,
    TaskType,
)
from health_analytics.schema import ColumnClassifier


class TestTaskInference:
    @pytest.fixture
    def inference(self) -> TaskInference:
        return TaskInference(max_classes=10)

    def test_binary_integers_are_classification(self, inference: TaskInference) -> None:
        assert inference.infer(pd.Series([0, 1, 0, 1, 1])) is TaskType.CLASSIFICATION

    def test_continuous_values_are_regression(self, inference: TaskInference) -> None:
        values = pd.Series(np.random.default_rng(0).normal(size=100))
        assert inference.infer(values) is TaskType.REGRESSION

    def test_string_labels_are_classification(self, inference: TaskInference) -> None:
        assert inference.infer(pd.Series(["yes", "no", "yes"])) is TaskType.CLASSIFICATION

    def test_many_distinct_integers_are_regression(self, inference: TaskInference) -> None:
        """A patient count spanning hundreds of values is a quantity, not a class."""
        assert inference.infer(pd.Series(range(500))) is TaskType.REGRESSION

    def test_class_limit_is_the_boundary(self) -> None:
        ten = pd.Series(list(range(10)) * 5)
        assert TaskInference(max_classes=10).infer(ten) is TaskType.CLASSIFICATION
        assert TaskInference(max_classes=9).infer(ten) is TaskType.REGRESSION

    def test_numeric_strings_are_parsed(self, inference: TaskInference) -> None:
        """Targets read from a text CSV arrive as strings and must still work."""
        assert inference.infer(pd.Series(["0", "1", "1", "0"])) is TaskType.CLASSIFICATION

    def test_prepare_target_casts_for_regression(self, inference: TaskInference) -> None:
        prepared = inference.prepare_target(pd.Series(["1.5", "2.5"]), TaskType.REGRESSION)
        assert prepared.dtype == float


class TestCategoricalEncoder:
    def test_missing_values_survive_encoding(self) -> None:
        """XGBoost needs to see NaN, not a code standing in for "unknown"."""
        frame = pd.DataFrame({"colour": ["red", None, "blue", "red"], "n": [1.0, 2.0, 3.0, 4.0]})
        encoded = CategoricalEncoder().fit_transform(frame, ["colour"])
        assert pd.isna(encoded.loc[1, "colour"])

    def test_distinct_categories_get_distinct_codes(self) -> None:
        frame = pd.DataFrame({"colour": ["red", "blue", "green", "red"]})
        encoded = CategoricalEncoder().fit_transform(frame, ["colour"])
        assert encoded["colour"].nunique() == 3
        assert encoded.loc[0, "colour"] == encoded.loc[3, "colour"]

    def test_column_order_is_preserved(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0], "cat": ["x", "y"], "b": [3.0, 4.0]})
        encoded = CategoricalEncoder().fit_transform(frame, ["cat"])
        assert list(encoded.columns) == ["a", "cat", "b"]

    def test_no_categorical_columns_is_a_passthrough(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        pd.testing.assert_frame_equal(CategoricalEncoder().fit_transform(frame, []), frame)


@pytest.fixture
def learnable_frame() -> pd.DataFrame:
    """Data with one feature that genuinely drives the outcome.

    ``signal`` determines the target; ``noise`` and ``category`` do not. Any
    importance method worth using must rank ``signal`` first.
    """
    rng = np.random.default_rng(7)
    n = 400
    signal = rng.normal(size=n)
    return pd.DataFrame(
        {
            "PAT_ID": range(n),
            "signal": signal,
            "noise": rng.normal(size=n),
            "category": rng.choice(["a", "b", "c"], n),
            "Y1": (signal > 0).astype(int),
            "Y2": signal * 3 + rng.normal(scale=0.1, size=n),
        }
    )


class TestFeatureImportanceStudy:
    def _run(self, frame: pd.DataFrame, target: str):
        classifier = ColumnClassifier()
        schema = classifier.classify(frame)
        config = ImportanceConfig(target_column=target, n_permutation_repeats=3)
        return FeatureImportanceStudy(config).run(classifier.coerce(frame, schema), schema)

    def test_classification_target_is_detected_and_learned(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        result = self._run(learnable_frame, "Y1")
        assert result.evaluation.task is TaskType.CLASSIFICATION
        assert result.evaluation.metrics["accuracy"] > 0.9

    def test_regression_target_is_detected_and_learned(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        result = self._run(learnable_frame, "Y2")
        assert result.evaluation.task is TaskType.REGRESSION
        assert result.evaluation.metrics["r2"] > 0.9

    def test_the_driving_feature_ranks_first(self, learnable_frame: pd.DataFrame) -> None:
        result = self._run(learnable_frame, "Y2")
        assert result.rankings.index[0] == "signal"

    def test_other_outcomes_are_excluded_from_the_features(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        """Y1 and Y2 both derive from `signal`; leaking one into the other
        would let the model score perfectly for the wrong reason."""
        result = self._run(learnable_frame, "Y2")
        assert "Y1" not in result.rankings.index

    def test_identifier_is_excluded_from_the_features(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        result = self._run(learnable_frame, "Y2")
        assert "PAT_ID" not in result.rankings.index

    def test_every_estimator_produces_a_column(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        result = self._run(learnable_frame, "Y2")
        assert {"gain", "permutation"} <= set(result.rankings.columns)

    def test_rows_with_a_missing_target_are_dropped(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        frame = learnable_frame.copy()
        frame.loc[0:49, "Y2"] = np.nan
        result = self._run(frame, "Y2")
        assert result.evaluation.n_train + result.evaluation.n_test == 350

    def test_unknown_target_raises_a_helpful_error(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        with pytest.raises(ValueError, match="not in dataset"):
            self._run(learnable_frame, "Y99")

    def test_fully_missing_target_raises(self, learnable_frame: pd.DataFrame) -> None:
        frame = learnable_frame.copy()
        frame["Y2"] = np.nan
        with pytest.raises(ValueError, match="No rows have an observed value"):
            self._run(frame, "Y2")

    def test_invalid_test_size_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="test_size"):
            ImportanceConfig(target_column="Y1", test_size=1.5)


class TestNonContiguousClassLabels:
    def test_arbitrary_class_codes_are_handled(self) -> None:
        """XGBoost requires labels 0..n-1; the real data uses codes like 3 and 7."""
        rng = np.random.default_rng(3)
        n = 300
        signal = rng.normal(size=n)
        frame = pd.DataFrame(
            {
                "signal": signal,
                "noise": rng.normal(size=n),
                "Y1": np.where(signal > 0, 7, 3),  # neither label is 0 or 1
            }
        )
        classifier = ColumnClassifier()
        schema = classifier.classify(frame)
        config = ImportanceConfig(target_column="Y1", n_permutation_repeats=3)
        result = FeatureImportanceStudy(config).run(classifier.coerce(frame, schema), schema)
        assert result.evaluation.metrics["accuracy"] > 0.9


class TestGainNameResolution:
    """Regression tests for the all-zero gain bug.

    XGBoost's booster reports positional keys (``f0``, ``f1``) when feature
    names were not retained. Looking those up by column name silently yields
    zero for every feature, which is exactly what the original script wrote to
    its output CSV for all 51 features.
    """

    def test_positional_keys_are_mapped_back_to_columns(self) -> None:
        resolved = GainImportance._resolve_names(
            {"f0": 12.5, "f2": 3.0}, ["alpha", "beta", "gamma"], None
        )
        assert resolved == {"alpha": 12.5, "gamma": 3.0}

    def test_named_keys_are_passed_through(self) -> None:
        raw = {"alpha": 1.0}
        assert GainImportance._resolve_names(raw, ["alpha"], ["alpha"]) is raw

    def test_out_of_range_indices_are_ignored(self) -> None:
        resolved = GainImportance._resolve_names({"f9": 1.0}, ["alpha"], None)
        assert resolved == {}

    def _rank(self, frame: pd.DataFrame):
        classifier = ColumnClassifier()
        schema = classifier.classify(frame)
        config = ImportanceConfig(target_column="Y2", n_permutation_repeats=3)
        return FeatureImportanceStudy(config).run(
            classifier.coerce(frame, schema), schema
        ).rankings

    def test_gain_is_not_uniformly_zero_end_to_end(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        assert (self._rank(learnable_frame)["gain"] > 0).any()

    def test_gain_and_permutation_agree_on_the_driving_feature(
        self, learnable_frame: pd.DataFrame
    ) -> None:
        rankings = self._rank(learnable_frame)
        assert rankings["gain"].idxmax() == "signal"
        assert rankings["permutation"].idxmax() == "signal"
