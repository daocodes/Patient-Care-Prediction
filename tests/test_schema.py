"""Tests for column type detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from health_analytics.config import SchemaConfig
from health_analytics.schema import ColumnClassifier


class TestNumericFraction:
    def test_all_numeric_strings_score_one(self, text_frame: pd.DataFrame) -> None:
        assert ColumnClassifier().numeric_fraction(text_frame["clean_numeric"]) == 1.0

    def test_text_column_scores_zero(self, text_frame: pd.DataFrame) -> None:
        assert ColumnClassifier().numeric_fraction(text_frame["categorical"]) == 0.0

    def test_one_bad_value_in_ten_scores_point_nine(self, text_frame: pd.DataFrame) -> None:
        assert ColumnClassifier().numeric_fraction(text_frame["mostly_numeric"]) == pytest.approx(0.9)

    def test_null_tokens_are_excluded_from_the_denominator(
        self, text_frame: pd.DataFrame
    ) -> None:
        # "", "nan" and "NA" are missing values, not unparseable numbers. If
        # they counted against the column it would score 0.7 and be misread as
        # categorical.
        assert ColumnClassifier().numeric_fraction(text_frame["with_null_tokens"]) == 1.0

    def test_empty_column_scores_zero(self) -> None:
        empty = pd.Series([np.nan, np.nan], dtype=object)
        assert ColumnClassifier().numeric_fraction(empty) == 0.0


class TestClassify:
    def test_threshold_is_inclusive(self, text_frame: pd.DataFrame) -> None:
        """A column at exactly the threshold counts as numeric."""
        classifier = ColumnClassifier(SchemaConfig(numeric_threshold=0.9))
        assert classifier.is_numeric(text_frame["mostly_numeric"])

    def test_stricter_threshold_rejects_the_same_column(
        self, text_frame: pd.DataFrame
    ) -> None:
        classifier = ColumnClassifier(SchemaConfig(numeric_threshold=0.95))
        assert not classifier.is_numeric(text_frame["mostly_numeric"])

    def test_every_column_lands_in_exactly_one_bucket(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        schema = ColumnClassifier().classify(mixed_frame)
        assert set(schema.numeric) | set(schema.categorical) == set(mixed_frame.columns)
        assert not set(schema.numeric) & set(schema.categorical)

    def test_identifiers_and_outcomes_are_recognised_by_name(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        schema = ColumnClassifier().classify(mixed_frame)
        assert schema.identifiers == ("PAT_ID",)
        assert set(schema.outcomes) == {"Y1", "Y2"}

    def test_all_empty_column_is_treated_as_categorical(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        """With no evidence, the tolerant path is the safe default."""
        schema = ColumnClassifier().classify(mixed_frame)
        assert "empty_col" in schema.categorical

    def test_invalid_threshold_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="numeric_threshold"):
            SchemaConfig(numeric_threshold=1.5)


class TestFeatures:
    def test_features_exclude_identifiers_and_outcomes(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        schema = ColumnClassifier().classify(mixed_frame)
        features = schema.features()
        assert "PAT_ID" not in features
        assert "Y1" not in features and "Y2" not in features
        assert "systolic" in features

    def test_explicit_exclusions_are_honoured(self, mixed_frame: pd.DataFrame) -> None:
        schema = ColumnClassifier().classify(mixed_frame)
        assert "systolic" not in schema.features(exclude=("systolic",))

    def test_numeric_features_are_a_subset_of_features(
        self, mixed_frame: pd.DataFrame
    ) -> None:
        schema = ColumnClassifier().classify(mixed_frame)
        assert set(schema.numeric_features()) <= set(schema.features())


class TestCoerce:
    def test_numeric_columns_get_numeric_dtypes(self, text_frame: pd.DataFrame) -> None:
        classifier = ColumnClassifier()
        schema = classifier.classify(text_frame)
        coerced = classifier.coerce(text_frame, schema)
        assert pd.api.types.is_numeric_dtype(coerced["clean_numeric"])

    def test_unparseable_values_become_nan(self, text_frame: pd.DataFrame) -> None:
        classifier = ColumnClassifier()
        schema = classifier.classify(text_frame)
        coerced = classifier.coerce(text_frame, schema)
        # "unknown" was the 10% that fell below the threshold.
        assert pd.isna(coerced.loc[4, "mostly_numeric"])

    def test_source_frame_is_not_mutated(self, text_frame: pd.DataFrame) -> None:
        classifier = ColumnClassifier()
        schema = classifier.classify(text_frame)
        before = text_frame.copy()
        classifier.coerce(text_frame, schema)
        pd.testing.assert_frame_equal(text_frame, before)
