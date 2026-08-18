"""Exploratory visual analysis.

Each plot is its own method so callers can run one without paying for the rest
-- t-SNE alone takes minutes on this dataset, and there is rarely a reason to
recompute it while iterating on a histogram.

Every method degrades rather than raising: a dataset without enough numeric
columns for PCA simply skips PCA. Exploration is the stage where you have the
least idea what the data looks like, so it is the worst place to abort a
20-minute run over one unsatisfied precondition.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import EDAConfig, PlotConfig
from .schema import DatasetSchema
from .visualization import FigureWriter, sns

logger = logging.getLogger(__name__)


class ExploratoryAnalysis:
    """Generates the standard set of exploratory figures for a dataset."""

    def __init__(
        self,
        writer: FigureWriter,
        config: EDAConfig | None = None,
        plot_config: PlotConfig | None = None,
    ) -> None:
        self._writer = writer
        self._config = config or EDAConfig()
        self._plots = plot_config or PlotConfig()

    def run_all(self, frame: pd.DataFrame, schema: DatasetSchema) -> list[str]:
        """Generate every figure, returning the names of those produced.

        Individual failures are logged and skipped rather than aborting the
        run, so one problematic plot does not cost you the other seven.
        """
        stages = (
            ("missingness", self.plot_missingness),
            ("distributions", self.plot_distributions),
            ("cluster map", self.plot_cluster_map),
            ("PCA", self.plot_pca),
            ("t-SNE", self.plot_tsne),
            ("pair plot", self.plot_pairs),
        )

        produced: list[str] = []
        for label, stage in stages:
            try:
                logger.info("Generating %s...", label)
                if stage(frame, schema):
                    produced.append(label)
            except Exception:  # noqa: BLE001 - one plot must not sink the run
                logger.exception("Skipping %s", label)
        return produced

    def plot_missingness(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Heatmap of which cells are missing.

        Reveals whether missingness is scattered at random or structured --
        blocks of adjacent rows or whole columns going dark usually mean a
        collection process changed partway through, which matters because it
        breaks the assumption imputation relies on.
        """
        with self._writer.figure(
            "missing_values_heatmap.png", size=(16, 10), title="Missing value matrix"
        ) as ax:
            sns.heatmap(
                frame.isna(),
                cbar=False,
                cmap=self._plots.sequential_cmap,
                yticklabels=False,
                ax=ax,
            )
        return True

    def plot_distributions(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Histograms for the highest-variance numeric columns.

        Ranked by variance because that is where the interesting shape is:
        near-constant columns produce a single bar and tell you nothing.
        """
        columns = self._top_by_variance(frame, schema, self._config.top_n_distributions)
        if not columns:
            return False

        cols = 3
        rows = int(np.ceil(len(columns) / cols))
        with self._writer.grid(
            "numeric_distributions.png", rows, cols, size=(20, 4 * rows)
        ) as axes:
            # strict=False is deliberate: the grid is rounded up to full rows,
            # so there are usually more axes than columns to draw into.
            for ax, column in zip(axes, columns, strict=False):
                sns.histplot(frame[column].dropna(), kde=True, ax=ax, color="steelblue")
                ax.set_title(column)
            # Blank out any unused cells in the last row.
            for ax in axes[len(columns):]:
                ax.set_visible(False)
        return True

    def plot_cluster_map(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Correlation heatmap with features reordered by hierarchical clustering.

        The clustering is what makes this more useful than a plain heatmap:
        related features end up adjacent, so groups of columns measuring the
        same underlying thing show up as blocks on the diagonal.
        """
        numeric = self._numeric_frame(frame, schema)
        if numeric.shape[1] < 2:
            return False

        sample = self._sample_rows(numeric, self._config.max_clustermap_rows)
        grid = sns.clustermap(
            sample.corr(),
            cmap=self._plots.diverging_cmap,
            center=0,
            linewidths=0.5,
            figsize=(15, 15),
        )
        # Title goes on the clustermap's own figure. The original code called
        # plt.title() here, which targeted the current Axes -- the colour bar --
        # and produced an untitled plot with a labelled legend.
        grid.figure.suptitle("Hierarchically clustered feature correlations", y=1.01)
        self._writer.save_existing(grid.figure, "hierarchical_cluster_map.png")
        return True

    def plot_pca(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Two-component PCA projection.

        A linear view of the global structure. The explained-variance figure in
        the title is the part to read: if two components capture only a few
        percent, the cloud's shape is close to meaningless.
        """
        scaled = self._scaled_matrix(frame, schema)
        if scaled is None or scaled.shape[1] < 3:
            return False

        pca = PCA(n_components=2, random_state=self._config.random_state)
        projected = pca.fit_transform(scaled)
        explained = pca.explained_variance_ratio_

        with self._writer.figure("pca_projection.png", size=(12, 10)) as ax:
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=10, c="teal")
            ax.set_title(f"PCA ({explained.sum():.1%} of variance explained)")
            ax.set_xlabel(f"PC1 ({explained[0]:.1%})")
            ax.set_ylabel(f"PC2 ({explained[1]:.1%})")
        return True

    def plot_tsne(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Two-component t-SNE projection.

        Complements PCA by preserving local neighbourhoods, so clusters PCA
        flattens together stay separate. Cost is superlinear in row count, so
        the input is subsampled -- with a seeded generator, unlike the original,
        which drew a fresh unseeded sample and so produced a different plot on
        every run.
        """
        scaled = self._scaled_matrix(frame, schema)
        if scaled is None or scaled.shape[1] < 3:
            return False

        limit = self._config.max_tsne_rows
        if len(scaled) > limit:
            rng = np.random.default_rng(self._config.random_state)
            scaled = scaled[rng.choice(len(scaled), limit, replace=False)]
            logger.info("Sampled %d rows for t-SNE", limit)

        # Perplexity is roughly "assumed neighbours per point" and must stay
        # below the sample size or sklearn raises.
        perplexity = min(self._config.tsne_perplexity, max(5, len(scaled) - 1))
        projected = TSNE(
            n_components=2,
            random_state=self._config.random_state,
            perplexity=perplexity,
        ).fit_transform(scaled)

        with self._writer.figure(
            "tsne_projection.png", size=(12, 10), title="t-SNE projection"
        ) as ax:
            ax.scatter(projected[:, 0], projected[:, 1], alpha=0.6, s=15, c="purple")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
        return True

    def plot_pairs(self, frame: pd.DataFrame, schema: DatasetSchema) -> bool:
        """Pairwise scatter matrix of the top numeric features.

        Coloured by a low-cardinality categorical column when one exists, which
        turns the plot into a quick check for whether that grouping separates
        the data at all.
        """
        columns = list(self._top_by_variance(frame, schema, self._config.top_n_pairplot))
        if len(columns) < 2:
            return False

        hue = self._pick_hue_column(frame, schema)
        sample = self._sample_rows(
            frame[columns + ([hue] if hue else [])], self._config.max_pairplot_rows
        )

        grid = sns.pairplot(
            sample, hue=hue, corner=True, diag_kind="kde", plot_kws={"alpha": 0.6}
        )
        self._writer.save_existing(grid.figure, "pairplot_top_features.png")
        return True

    # ---- shared helpers -------------------------------------------------

    def _numeric_frame(self, frame: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
        """Numeric feature columns, median-filled, with constants dropped."""
        columns = [c for c in schema.numeric_features() if c in frame.columns]
        if not columns:
            return pd.DataFrame(index=frame.index)
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.fillna(numeric.median())
        return numeric.loc[:, numeric.std() > 0]

    def _scaled_matrix(self, frame: pd.DataFrame, schema: DatasetSchema) -> np.ndarray | None:
        """Standardised numeric matrix for the projection methods.

        Both PCA and t-SNE are variance-driven, so an unscaled column measured
        in thousands would define the projection on its own.
        """
        numeric = self._numeric_frame(frame, schema)
        if numeric.empty or numeric.shape[1] < 2:
            return None
        return StandardScaler().fit_transform(numeric)

    def _top_by_variance(
        self, frame: pd.DataFrame, schema: DatasetSchema, n: int
    ) -> tuple[str, ...]:
        numeric = self._numeric_frame(frame, schema)
        if numeric.empty:
            return ()
        return tuple(numeric.var().sort_values(ascending=False).head(n).index)

    def _sample_rows(self, frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Take a seeded sample when ``frame`` exceeds ``limit`` rows."""
        if len(frame) <= limit:
            return frame
        return frame.sample(limit, random_state=self._config.random_state)

    def _pick_hue_column(self, frame: pd.DataFrame, schema: DatasetSchema) -> str | None:
        """First categorical column with few enough levels to colour by."""
        for column in schema.categorical:
            if column in frame.columns and 1 < frame[column].nunique() <= self._config.max_hue_cardinality:
                return column
        return None
