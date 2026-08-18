"""Missing-value imputation.

Two tracks, because the dataset mixes measurements with labels:

* **Numeric** columns go through KNN imputation on standardised values. A
  patient's missing blood pressure is better estimated from patients who
  resemble them across every other vital than from the column median.
* **Categorical** columns get the most frequent value. There is no meaningful
  distance between "sunny" and "rain" to average over.

Both tracks are followed by a deterministic fallback so the output is
guaranteed complete: KNN leaves NaN in place when a row has no usable
neighbours, and the mode is undefined for a column that is entirely empty.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from .config import ImputationConfig
from .schema import ColumnClassifier, DatasetSchema


@dataclass
class ImputationReport:
    """Record of what an imputation run changed.

    Returned alongside the imputed frame rather than printed, so callers can
    assert on it in tests or serialise it next to the output file.
    """

    filled_by_column: dict[str, int] = field(default_factory=dict)
    strategy_by_column: dict[str, str] = field(default_factory=dict)
    neighbors_used: int | None = None

    @property
    def total_filled(self) -> int:
        return sum(self.filled_by_column.values())

    def record(self, column: str, count: int, strategy: str) -> None:
        """Add ``count`` filled cells for ``column`` under ``strategy``."""
        if count <= 0:
            return
        self.filled_by_column[column] = self.filled_by_column.get(column, 0) + count
        self.strategy_by_column[column] = strategy

    def summary(self) -> str:
        """Human-readable digest, ordered by how much each column was filled."""
        if not self.filled_by_column:
            return "No missing values were filled."
        ranked = sorted(self.filled_by_column.items(), key=lambda kv: kv[1], reverse=True)
        lines = [f"Filled {self.total_filled:,} cells across {len(ranked)} columns:"]
        for column, count in ranked[:15]:
            lines.append(f"  {column:<20} {count:>8,}  ({self.strategy_by_column[column]})")
        if len(ranked) > 15:
            lines.append(f"  ... and {len(ranked) - 15} more columns")
        return "\n".join(lines)


class ColumnImputer(ABC):
    """Fills missing values for a subset of a frame's columns.

    Implementations receive only the columns they are responsible for and
    return a frame with the same index and columns. Keeping the contract this
    narrow is what lets :class:`ImputationPipeline` run them in sequence
    without knowing anything about how they work.
    """

    #: Short label recorded in the report for columns this imputer touched.
    name: str = "unknown"

    @abstractmethod
    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return ``frame`` with missing values filled."""


class KNNNumericImputer(ColumnImputer):
    """K-nearest-neighbours imputation for numeric columns.

    Standardises before imputing and reverses the transform afterwards. Without
    that step the neighbour distance is dominated by whichever column happens
    to have the largest units -- here that is a cost field in the tens of
    thousands, which would otherwise drown out every vital sign.
    """

    name = "knn"

    def __init__(self, config: ImputationConfig) -> None:
        self._config = config
        self.neighbors_used: int | None = None

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or frame.shape[1] == 0:
            return frame.copy()

        numeric = frame.apply(pd.to_numeric, errors="coerce")
        missing_mask = numeric.isna()

        # Columns with no observed values at all have no median and no
        # neighbours to learn from. Hold them aside; the pipeline's fallback
        # handles them explicitly rather than letting them poison the distance
        # metric with all-NaN rows.
        observed = [c for c in numeric.columns if numeric[c].notna().any()]
        if not observed:
            return numeric

        usable = numeric[observed]

        # Temporary median fill exists only so the scaler can estimate a mean
        # and standard deviation. The NaNs are restored immediately afterwards
        # so KNNImputer still sees genuine gaps.
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(usable.fillna(usable.median())),
            index=usable.index,
            columns=usable.columns,
        )
        scaled[usable.isna()] = np.nan

        # k cannot exceed the number of other rows available to borrow from.
        self.neighbors_used = min(self._config.n_neighbors, max(1, len(scaled) - 1))

        knn = KNNImputer(
            n_neighbors=self.neighbors_used,
            weights=self._config.weights,
            metric="nan_euclidean",
        )
        imputed_scaled = knn.fit_transform(scaled)

        result = numeric.copy()
        result[observed] = scaler.inverse_transform(imputed_scaled)

        # Restore integer-valued columns to integers. Averaging neighbours turns
        # a count of prior visits into 3.0000000004; writing that to the output
        # file makes a discrete field look continuous.
        for column in observed:
            result[column] = self._restore_integrality(
                result[column], was_missing=missing_mask[column]
            )
        return result

    @staticmethod
    def _restore_integrality(values: pd.Series, was_missing: pd.Series) -> pd.Series:
        """Round to integers if every originally-observed value was an integer.

        Judged on the observed values only. The imputed ones are fractional by
        construction, so including them would mean no column ever qualifies.
        """
        observed = values[~was_missing].dropna()
        if observed.empty or not np.all(np.modf(observed.to_numpy(dtype=float))[0] == 0):
            return values.astype(float)
        rounded = values.round()
        # Nullable Int64 preserves any NaN the imputer could not resolve;
        # plain int would raise on them.
        return rounded.astype("Int64") if rounded.isna().any() else rounded.astype("int64")


class ModeCategoricalImputer(ColumnImputer):
    """Most-frequent-value imputation for categorical columns."""

    name = "mode"

    def __init__(self, config: ImputationConfig) -> None:
        self._config = config

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or frame.shape[1] == 0:
            return frame.copy()

        result = frame.copy()
        # SimpleImputer raises on a column with nothing to take a mode from, so
        # those columns are separated out and given the explicit placeholder.
        has_values = [c for c in result.columns if result[c].notna().any()]
        empty = [c for c in result.columns if c not in has_values]

        if has_values:
            imputer = SimpleImputer(strategy="most_frequent")
            result[has_values] = imputer.fit_transform(result[has_values])

        for column in empty:
            result[column] = self._config.categorical_placeholder
        return result


class ImputationPipeline:
    """Runs both imputers over a frame and guarantees a complete result.

    The pipeline owns three things the individual imputers deliberately do not:
    routing columns by type, applying the fallbacks, and asserting that rows
    were neither added, dropped, nor reordered.
    """

    def __init__(
        self,
        config: ImputationConfig | None = None,
        classifier: ColumnClassifier | None = None,
    ) -> None:
        self._config = config or ImputationConfig()
        self._classifier = classifier or ColumnClassifier()

    def run(
        self, frame: pd.DataFrame, schema: DatasetSchema | None = None
    ) -> tuple[pd.DataFrame, ImputationReport]:
        """Impute ``frame`` and return the result with a report.

        Args:
            frame: Raw data, ideally read as text so nothing has been coerced.
            schema: Column roles. Detected from ``frame`` when omitted.

        Returns:
            The imputed frame (same index and column order as the input) and an
            :class:`ImputationReport` describing every fill.
        """
        schema = schema or self._classifier.classify(frame)
        report = ImputationReport()

        original = frame.copy(deep=True)
        result = self._classifier.coerce(frame, schema)

        skip = set(self._config.skip_columns)
        numeric_cols = [c for c in schema.numeric if c not in skip]
        categorical_cols = [c for c in schema.categorical if c not in skip]

        missing_before = result.isna()

        if numeric_cols:
            knn = KNNNumericImputer(self._config)
            result[numeric_cols] = knn.fit_transform(result[numeric_cols])
            report.neighbors_used = knn.neighbors_used
            self._record_fills(report, missing_before, result, numeric_cols, knn.name)

        if categorical_cols:
            mode = ModeCategoricalImputer(self._config)
            result[categorical_cols] = mode.fit_transform(result[categorical_cols])
            self._record_fills(report, missing_before, result, categorical_cols, mode.name)

        self._apply_fallbacks(result, report, numeric_cols, categorical_cols)

        if self._config.keep_original_columns:
            for column in numeric_cols + categorical_cols:
                result[f"{column}_orig"] = original[column]

        self._assert_shape_preserved(original, result)
        return self._restore_column_order(original, result), report

    @staticmethod
    def _record_fills(
        report: ImputationReport,
        missing_before: pd.DataFrame,
        result: pd.DataFrame,
        columns: list[str],
        strategy: str,
    ) -> None:
        """Count cells that were missing before the stage and are filled now."""
        for column in columns:
            filled = int((missing_before[column] & result[column].notna()).sum())
            report.record(column, filled, strategy)

    def _apply_fallbacks(
        self,
        result: pd.DataFrame,
        report: ImputationReport,
        numeric_cols: list[str],
        categorical_cols: list[str],
    ) -> None:
        """Fill anything the primary imputers could not resolve.

        Mutates ``result`` in place. KNN leaves a value missing when a row has
        no neighbour with that column observed, and an all-empty categorical
        column has no mode; both are handled here so callers can rely on the
        output having no gaps outside ``skip_columns``.
        """
        for column in numeric_cols:
            gaps = result[column].isna()
            if not gaps.any():
                continue
            median = result[column].median()
            # An entirely-empty numeric column has no median either; zero is the
            # only value available, and the report makes the substitution visible.
            fill_value = 0 if pd.isna(median) else median
            result.loc[gaps, column] = fill_value
            report.record(column, int(gaps.sum()), "median-fallback")

        for column in categorical_cols:
            gaps = result[column].isna() | (
                result[column].astype("string").str.strip() == ""
            )
            if not gaps.any():
                continue
            observed = result.loc[~gaps, column].dropna()
            mode = observed.mode()
            fill_value = (
                mode.iloc[0] if not mode.empty else self._config.categorical_placeholder
            )
            result.loc[gaps, column] = fill_value
            report.record(column, int(gaps.sum()), "mode-fallback")

    @staticmethod
    def _assert_shape_preserved(original: pd.DataFrame, result: pd.DataFrame) -> None:
        """Guard against silently losing or reordering patient records.

        Imputation must be a cell-level operation. A row count or index that
        changed means a bug upstream, and the resulting file could no longer be
        joined back to anything.
        """
        if len(result) != len(original):
            raise RuntimeError(
                f"Row count changed during imputation: "
                f"{len(original)} -> {len(result)}"
            )
        if not result.index.equals(original.index):
            raise RuntimeError("Row order changed during imputation")

    @staticmethod
    def _restore_column_order(
        original: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        """Put columns back in their input order, with any backups appended."""
        primary = [c for c in original.columns if c in result.columns]
        extra = [c for c in result.columns if c not in primary]
        return result[primary + extra]
