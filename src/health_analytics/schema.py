"""Column type detection.

Four of the five original scripts carried their own copy of a "is this column
numeric?" helper, and they disagreed: one required 90% of values to parse, one
required 50%, and one accepted a single parseable value. Depending on which
script you ran, the same column was numeric or categorical. This module is the
single implementation everything else defers to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import SchemaConfig

# Strings Excel and CSV exports use for "no value". Read with ``dtype=object``
# these arrive as literal text rather than NaN, and would otherwise make an
# entire numeric column look categorical.
_NULL_TOKENS = ("", "nan", "NaN", "NA", "N/A", "null", "NULL", "None", "-", "--")


@dataclass(frozen=True)
class DatasetSchema:
    """The resolved column roles for one dataset.

    Attributes are tuples rather than lists so a schema can be safely shared
    between stages without one of them mutating it.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    identifiers: tuple[str, ...]
    outcomes: tuple[str, ...]

    @property
    def all_columns(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    def features(self, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        """Columns usable as model inputs.

        Identifiers and outcomes are dropped: a patient ID is numeric and
        highly predictive of nothing, and leaving other outcome columns in
        while predicting one of them leaks the answer.
        """
        blocked = set(self.identifiers) | set(self.outcomes) | set(exclude)
        return tuple(c for c in self.all_columns if c not in blocked)

    def numeric_features(self, exclude: tuple[str, ...] = ()) -> tuple[str, ...]:
        allowed = set(self.features(exclude))
        return tuple(c for c in self.numeric if c in allowed)

    def describe(self) -> str:
        """One-line summary for logs."""
        return (
            f"{len(self.numeric)} numeric, {len(self.categorical)} categorical, "
            f"{len(self.identifiers)} identifier, {len(self.outcomes)} outcome"
        )


class ColumnClassifier:
    """Splits a frame's columns into numeric and categorical.

    The test is deliberately tolerant. A column counts as numeric when at least
    ``numeric_threshold`` of its non-null values parse as numbers, so a vitals
    field holding a few free-text entries is still analysed as a number rather
    than being demoted to a 90,000-category string column.
    """

    def __init__(self, config: SchemaConfig | None = None) -> None:
        self._config = config or SchemaConfig()

    @staticmethod
    def normalize_nulls(series: pd.Series) -> pd.Series:
        """Replace placeholder null tokens with real NaN.

        Applied before any parsing so that ``""`` and ``"nan"`` are counted as
        missing rather than as unparseable values that drag a column below the
        numeric threshold.
        """
        if series.dtype == object:
            return series.replace(list(_NULL_TOKENS), np.nan)
        return series

    def numeric_fraction(self, series: pd.Series) -> float:
        """Fraction of non-null values that parse as numbers, in ``[0, 1]``.

        An all-null column returns ``0.0``: with no evidence either way, the
        safe assumption is categorical, since the categorical path tolerates
        numbers but the numeric path would coerce real labels to NaN.
        """
        non_null = self.normalize_nulls(series).dropna()
        if non_null.empty:
            return 0.0
        parsed = pd.to_numeric(non_null, errors="coerce")
        return float(parsed.notna().sum()) / float(len(non_null))

    def is_numeric(self, series: pd.Series) -> bool:
        return self.numeric_fraction(series) >= self._config.numeric_threshold

    def classify(self, frame: pd.DataFrame) -> DatasetSchema:
        """Assign every column in ``frame`` to exactly one role.

        Identifier and outcome columns are recognised by name from the config
        and reported separately, but they are *also* type-classified, so a
        numeric outcome still appears in ``schema.numeric`` and can be
        correlated or plotted like any other number.
        """
        numeric: list[str] = []
        categorical: list[str] = []

        for column in frame.columns:
            if self.is_numeric(frame[column]):
                numeric.append(column)
            else:
                categorical.append(column)

        present = set(frame.columns)
        return DatasetSchema(
            numeric=tuple(numeric),
            categorical=tuple(categorical),
            identifiers=tuple(c for c in self._config.id_columns if c in present),
            outcomes=tuple(c for c in self._config.outcome_columns if c in present),
        )

    def coerce(self, frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
        """Return a copy of ``frame`` with numeric columns given numeric dtypes.

        Values that fail to parse become NaN, which is the intended outcome:
        they were the minority that fell under the threshold, and downstream
        stages treat them as missing rather than guessing at a value.
        """
        coerced = frame.copy()
        for column in schema.numeric:
            coerced[column] = pd.to_numeric(
                self.normalize_nulls(coerced[column]), errors="coerce"
            )
        for column in schema.categorical:
            coerced[column] = self.normalize_nulls(coerced[column])
        return coerced
