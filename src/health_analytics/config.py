"""Configuration objects for the analysis pipeline.

Every stage takes a frozen dataclass rather than reading module-level globals.
That keeps stages independently testable (build a config in a test, no files or
environment needed) and makes the knobs discoverable in one place instead of
scattered across ``CONFIG`` blocks at the top of five different scripts.
"""

from __future__ import annotations

from dataclasses import dataclass

# A column is treated as numeric when at least this fraction of its non-null
# values survive `pd.to_numeric`. Clinical exports routinely carry a handful of
# free-text entries ("<0.1", "pending") in otherwise numeric fields, so an
# all-or-nothing test would misclassify most of the vitals columns.
DEFAULT_NUMERIC_THRESHOLD = 0.9

# Seed used anywhere a stage samples, splits, or initialises a model. Fixed so
# that two runs over the same input produce byte-identical figures and tables.
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True)
class SchemaConfig:
    """Rules for deciding which columns count as numeric.

    ``id_columns`` and ``outcome_columns`` are excluded from feature matrices.
    They are named rather than inferred because a patient identifier is
    perfectly numeric and would otherwise sail through detection and end up
    as the most "important" feature in the model.
    """

    numeric_threshold: float = DEFAULT_NUMERIC_THRESHOLD
    id_columns: tuple[str, ...] = ("PAT_ID",)
    outcome_columns: tuple[str, ...] = ("Y1", "Y2", "Y3", "Y4", "Y5")

    def __post_init__(self) -> None:
        if not 0.0 < self.numeric_threshold <= 1.0:
            raise ValueError(
                f"numeric_threshold must be in (0, 1], got {self.numeric_threshold}"
            )


@dataclass(frozen=True)
class ImputationConfig:
    """Settings for the two-track imputation pipeline.

    ``n_neighbors`` is a request, not a guarantee: it is capped at
    ``n_samples - 1`` at fit time so that the pipeline still runs on the small
    frames used in tests.
    """

    n_neighbors: int = 20
    weights: str = "distance"
    categorical_placeholder: str = "MISSING"
    # Keep a verbatim copy of each imputed column as ``<name>_orig``. Useful for
    # auditing what the imputer changed; doubles the width of the output file.
    keep_original_columns: bool = False
    # Columns to pass through untouched, e.g. fields whose missingness is itself
    # meaningful and must not be filled in.
    skip_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {self.n_neighbors}")


@dataclass(frozen=True)
class EDAConfig:
    """Limits that keep exploratory plots readable and fast.

    The dataset has 70 columns and 100k rows. Plotting all of it produces
    unreadable figures and multi-hour t-SNE runs, so each plot takes the top-N
    columns by variance and samples rows where the algorithm is superlinear.
    """

    top_n_distributions: int = 12
    top_n_pairplot: int = 4
    max_pairplot_rows: int = 1_000
    max_clustermap_rows: int = 1_000
    max_tsne_rows: int = 2_000
    tsne_perplexity: int = 30
    max_hue_cardinality: int = 10
    random_state: int = DEFAULT_RANDOM_STATE


@dataclass(frozen=True)
class ImportanceConfig:
    """Settings for the supervised feature-importance study."""

    target_column: str
    test_size: float = 0.20
    random_state: int = DEFAULT_RANDOM_STATE
    n_permutation_repeats: int = 10
    top_n_features: int = 30
    max_shap_rows: int = 2_000
    # A numeric target with integer-valued entries and no more than this many
    # distinct values is treated as classification rather than regression.
    max_classification_classes: int = 10

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(f"test_size must be in (0, 1), got {self.test_size}")


@dataclass(frozen=True)
class PlotConfig:
    """Shared matplotlib/seaborn styling.

    Centralised so every figure the project emits looks like it came from the
    same report rather than from five scripts with five different themes.
    """

    style: str = "whitegrid"
    context: str = "talk"
    figure_size: tuple[float, float] = (12.0, 8.0)
    heatmap_size: tuple[float, float] = (14.0, 12.0)
    dpi: int = 150
    diverging_cmap: str = "vlag"
    sequential_cmap: str = "viridis"
