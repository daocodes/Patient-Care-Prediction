"""Text profiling: the quick pass you run before plotting anything.

Answers the four questions worth asking of an unfamiliar table -- what is
missing, what is redundant, what is skewed, and what is effectively constant --
and returns them as data rather than printing them, so the results can be
asserted on in tests or written to a file.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .correlation import CorrelationAnalyzer
from .schema import DatasetSchema

# A categorical column where one value covers this much of the data carries
# almost no signal, and can quietly break a stratified split.
DOMINANCE_THRESHOLD = 0.90

# Beyond this, a distribution is skewed enough that a log transform is usually
# worth trying before any model that assumes symmetry.
SKEW_THRESHOLD = 1.0


@dataclass(frozen=True)
class DatasetProfile:
    """Summary statistics for one dataset."""

    n_rows: int
    n_columns: int
    schema: DatasetSchema
    missing_fraction: pd.Series
    top_correlations: pd.DataFrame
    skewness: pd.Series
    dominant_categories: dict[str, tuple[str, float]]

    @property
    def columns_with_missing(self) -> pd.Series:
        return self.missing_fraction[self.missing_fraction > 0]

    @property
    def highly_skewed(self) -> pd.Series:
        return self.skewness[self.skewness.abs() >= SKEW_THRESHOLD]

    def to_text(self, top_n: int = 10) -> str:
        """Render the profile as a readable report."""
        sections = [
            f"Dataset: {self.n_rows:,} rows x {self.n_columns} columns",
            f"Schema:  {self.schema.describe()}",
            "",
            self._section_missing(top_n),
            "",
            self._section_correlations(top_n),
            "",
            self._section_skew(top_n),
            "",
            self._section_dominance(),
        ]
        return "\n".join(sections)

    def _section_missing(self, top_n: int) -> str:
        missing = self.columns_with_missing
        if missing.empty:
            return "MISSING VALUES\n  None."
        lines = [f"MISSING VALUES ({len(missing)} columns affected)"]
        for column, fraction in missing.head(top_n).items():
            lines.append(f"  {column:<20} {fraction:>7.2%}")
        return "\n".join(lines)

    def _section_correlations(self, top_n: int) -> str:
        if self.top_correlations.empty:
            return "STRONGEST CORRELATIONS\n  Not enough numeric columns."
        lines = ["STRONGEST CORRELATIONS"]
        for row in self.top_correlations.head(top_n).itertuples():
            lines.append(
                f"  {row.feature_1:<18} {row.feature_2:<18} {row.correlation:>+7.3f}"
            )
        return "\n".join(lines)

    def _section_skew(self, top_n: int) -> str:
        skewed = self.highly_skewed
        if skewed.empty:
            return f"SKEWNESS\n  No column exceeds |skew| = {SKEW_THRESHOLD}."
        lines = [f"SKEWNESS (|skew| >= {SKEW_THRESHOLD}, consider a transform)"]
        for column, value in skewed.head(top_n).items():
            lines.append(f"  {column:<20} {value:>+8.2f}")
        return "\n".join(lines)

    def _section_dominance(self) -> str:
        if not self.dominant_categories:
            return f"CATEGORICAL DOMINANCE\n  No column exceeds {DOMINANCE_THRESHOLD:.0%}."
        lines = [f"CATEGORICAL DOMINANCE (one value covers >= {DOMINANCE_THRESHOLD:.0%})"]
        for column, (value, share) in self.dominant_categories.items():
            lines.append(f"  {column:<20} '{value}' {share:>7.1%}")
        return "\n".join(lines)


class DatasetProfiler:
    """Builds a :class:`DatasetProfile` from a frame."""

    def __init__(self, correlation_analyzer: CorrelationAnalyzer | None = None) -> None:
        self._correlations = correlation_analyzer or CorrelationAnalyzer()

    def profile(self, frame: pd.DataFrame, schema: DatasetSchema) -> DatasetProfile:
        """Compute the full profile.

        Correlations are computed defensively: a frame with fewer than two
        usable numeric columns yields an empty table rather than an error, so
        profiling works on any input.
        """
        numeric_columns = [c for c in schema.numeric_features() if c in frame.columns]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")

        try:
            top_correlations = self._correlations.analyze(frame, schema).ranked_pairs
        except ValueError:
            top_correlations = pd.DataFrame(
                columns=["feature_1", "feature_2", "correlation", "abs_correlation"]
            )

        return DatasetProfile(
            n_rows=len(frame),
            n_columns=frame.shape[1],
            schema=schema,
            missing_fraction=frame.isna().mean().sort_values(ascending=False),
            top_correlations=top_correlations,
            skewness=(
                numeric.skew().sort_values(key=abs, ascending=False)
                if not numeric.empty
                else pd.Series(dtype=float)
            ),
            dominant_categories=self._find_dominant(frame, schema),
        )

    @staticmethod
    def _find_dominant(
        frame: pd.DataFrame, schema: DatasetSchema
    ) -> dict[str, tuple[str, float]]:
        """Categorical columns where a single value covers almost everything."""
        dominant: dict[str, tuple[str, float]] = {}
        for column in schema.categorical:
            if column not in frame.columns:
                continue
            counts = frame[column].value_counts(normalize=True, dropna=True)
            # An all-null column yields empty counts. The original indexed
            # `.iloc[0]` unconditionally here and raised IndexError on the
            # first fully-empty column it met.
            if counts.empty:
                continue
            if counts.iloc[0] >= DOMINANCE_THRESHOLD:
                dominant[column] = (str(counts.index[0]), float(counts.iloc[0]))
        return dominant
