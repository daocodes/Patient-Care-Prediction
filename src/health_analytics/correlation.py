"""Pairwise correlation analysis.

Produces the full matrix, a ranked list of the strongest pairs, and heatmaps.
Pearson and Spearman are both offered because they answer different questions:
Pearson finds linear relationships, Spearman finds monotone ones, and a pair
that scores high on the second but not the first is usually a curved
relationship worth looking at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .config import PlotConfig
from .schema import DatasetSchema
from .visualization import FigureWriter, sns

CorrelationMethod = Literal["pearson", "spearman", "kendall"]


@dataclass(frozen=True)
class CorrelationResult:
    """A correlation matrix plus its ranked pair list."""

    method: CorrelationMethod
    matrix: pd.DataFrame
    ranked_pairs: pd.DataFrame

    def top(self, n: int = 10) -> pd.DataFrame:
        """The ``n`` most strongly correlated pairs, by absolute value."""
        return self.ranked_pairs.head(n)

    def above(self, threshold: float) -> pd.DataFrame:
        """Pairs whose absolute correlation meets ``threshold``.

        Useful for flagging redundant features before modelling.
        """
        return self.ranked_pairs[self.ranked_pairs["abs_correlation"] >= threshold]


class CorrelationAnalyzer:
    """Computes correlations over the numeric columns of a frame."""

    def __init__(self, plot_config: PlotConfig | None = None) -> None:
        self._plots = plot_config or PlotConfig()

    def analyze(
        self,
        frame: pd.DataFrame,
        schema: DatasetSchema,
        method: CorrelationMethod = "pearson",
        exclude_identifiers: bool = True,
    ) -> CorrelationResult:
        """Correlate every numeric column against every other.

        Args:
            frame: Data with numeric columns already coerced to numeric dtypes.
            schema: Column roles, used to select the numeric columns.
            method: ``pearson``, ``spearman``, or ``kendall``.
            exclude_identifiers: Drop patient IDs. A sequential ID correlates
                with anything that drifted over the collection period, which is
                an artefact of row order rather than a finding.

        Raises:
            ValueError: If fewer than two numeric columns are available.
        """
        columns = self._select_columns(schema, exclude_identifiers)
        available = [c for c in columns if c in frame.columns]
        if len(available) < 2:
            raise ValueError(
                f"Need at least 2 numeric columns to correlate, found {len(available)}"
            )

        numeric = frame[available].apply(pd.to_numeric, errors="coerce")
        # Constant columns produce NaN correlations and a band of blank cells
        # across the heatmap. They carry no information, so drop them.
        varying = numeric.loc[:, numeric.std(numeric_only=True) > 0]

        matrix = varying.corr(method=method)
        return CorrelationResult(
            method=method,
            matrix=matrix,
            ranked_pairs=self.rank_pairs(matrix),
        )

    @staticmethod
    def _select_columns(schema: DatasetSchema, exclude_identifiers: bool) -> tuple[str, ...]:
        if exclude_identifiers:
            return tuple(c for c in schema.numeric if c not in schema.identifiers)
        return schema.numeric

    @staticmethod
    def rank_pairs(matrix: pd.DataFrame) -> pd.DataFrame:
        """Flatten a correlation matrix into unique pairs, strongest first.

        A correlation matrix is symmetric with a unit diagonal, so two thirds of
        it is noise: every pair appears twice and every column correlates
        perfectly with itself. Masking to the upper triangle removes both in one
        step. The original implementation instead built a sorted tuple per row
        with ``DataFrame.apply`` and dropped duplicates, which is O(n^2) Python
        calls -- around 4,900 for this dataset's 70 columns, against a single
        vectorised NumPy operation here.
        """
        # k=1 excludes the diagonal along with the lower triangle. Indexing by
        # the mask's coordinates rather than masking and calling `.stack()`
        # keeps this independent of pandas' NaN-dropping behaviour, which
        # changed in pandas 3.0 and would otherwise let the masked half through.
        rows, cols = np.where(np.triu(np.ones(matrix.shape, dtype=bool), k=1))
        pairs = pd.DataFrame(
            {
                "feature_1": matrix.index[rows],
                "feature_2": matrix.columns[cols],
                "correlation": matrix.to_numpy()[rows, cols],
            }
        )
        # A NaN correlation means one side was constant over the rows the pair
        # shared; there is no relationship to rank.
        pairs = pairs.dropna(subset=["correlation"])
        pairs["abs_correlation"] = pairs["correlation"].abs()
        return pairs.sort_values("abs_correlation", ascending=False).reset_index(drop=True)

    def plot_heatmap(
        self,
        result: CorrelationResult,
        writer: FigureWriter,
        filename: str | None = None,
        annotate_threshold: int = 25,
    ) -> None:
        """Draw a triangular heatmap of the correlation matrix.

        Only the lower triangle is drawn, for the same reason the ranked list
        is deduplicated: the mirrored half adds no information. Cell values are
        printed only when the matrix is small enough for them to be legible.
        """
        filename = filename or f"correlation_{result.method}.png"
        matrix = result.matrix
        mask = np.triu(np.ones_like(matrix, dtype=bool))
        n = len(matrix)

        with writer.figure(
            filename,
            size=self._plots.heatmap_size,
            title=f"{result.method.capitalize()} correlation",
        ) as ax:
            sns.heatmap(
                matrix,
                mask=mask,
                cmap=self._plots.diverging_cmap,
                vmin=-1,
                vmax=1,
                center=0,
                square=True,
                linewidths=0.5,
                annot=n <= annotate_threshold,
                fmt=".2f",
                # Labels become an unreadable smear past roughly 40 columns.
                xticklabels=n <= 40,
                yticklabels=n <= 40,
                cbar_kws={"shrink": 0.5},
                ax=ax,
            )
