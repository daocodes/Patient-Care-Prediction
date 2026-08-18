"""Command-line entry point.

One command with five subcommands, replacing five scripts that each had to be
edited to change their input path. Every stage shares the same ``--input`` and
``--output-dir`` handling, so switching datasets never means touching source.

    health-analytics profile   --input data/encounters.csv
    health-analytics impute    --input data/raw.xlsx --output data/imputed.xlsx
    health-analytics eda       --input data/encounters.csv
    health-analytics correlate --input data/encounters.csv --method spearman
    health-analytics importance --input data/encounters.csv --target Y2
    health-analytics validate  --input data/imputed.xlsx --original data/raw.xlsx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    EDAConfig,
    ImportanceConfig,
    ImputationConfig,
    PlotConfig,
    SchemaConfig,
)
from .correlation import CorrelationAnalyzer
from .datasets import DataSourceError, source_for, write_frame
from .eda import ExploratoryAnalysis
from .importance import FeatureImportanceStudy
from .imputation import ImputationPipeline
from .profiling import DatasetProfiler
from .schema import ColumnClassifier
from .validation import ImputationValidator
from .visualization import FigureWriter, apply_style

logger = logging.getLogger("health_analytics")


def build_parser() -> argparse.ArgumentParser:
    """Assemble the argument parser and its subcommands."""
    parser = argparse.ArgumentParser(
        prog="health-analytics",
        description="Analysis pipeline for clinical encounter data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Emit debug-level logging."
    )

    # Options every subcommand needs, attached to each in turn. Defined as a
    # parent parser so the flags stay identical across subcommands rather than
    # drifting apart as new ones are added.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--input", type=Path, required=True, help="Path to a .csv or .xlsx dataset."
    )
    common.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory for figures and tables (default: output).",
    )
    common.add_argument(
        "--numeric-threshold",
        type=float,
        default=SchemaConfig.numeric_threshold,
        help="Fraction of values that must parse as numbers for a column to "
        "count as numeric (default: %(default)s).",
    )

    subcommands = parser.add_subparsers(dest="command", required=True)

    profile = subcommands.add_parser(
        "profile", parents=[common], help="Print a text summary of the dataset."
    )
    profile.add_argument(
        "--save", action="store_true", help="Also write the summary to a file."
    )
    profile.set_defaults(handler=run_profile)

    impute = subcommands.add_parser(
        "impute", parents=[common], help="Fill missing values (KNN + mode)."
    )
    impute.add_argument(
        "--output", type=Path, required=True, help="Path for the imputed dataset."
    )
    impute.add_argument(
        "--neighbors",
        type=int,
        default=ImputationConfig.n_neighbors,
        help="KNN neighbours, capped at n_rows-1 (default: %(default)s).",
    )
    impute.add_argument(
        "--skip",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Columns to leave untouched.",
    )
    impute.add_argument(
        "--keep-originals",
        action="store_true",
        help="Append each source column as <name>_orig for auditing.",
    )
    impute.set_defaults(handler=run_impute)

    eda = subcommands.add_parser(
        "eda", parents=[common], help="Generate exploratory figures."
    )
    eda.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE, which dominates the runtime on large inputs.",
    )
    eda.set_defaults(handler=run_eda)

    correlate = subcommands.add_parser(
        "correlate", parents=[common], help="Compute and rank feature correlations."
    )
    correlate.add_argument(
        "--method",
        choices=["pearson", "spearman", "kendall"],
        default="pearson",
        help="Correlation method (default: %(default)s).",
    )
    correlate.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Report pairs at or above this absolute correlation "
        "(default: %(default)s).",
    )
    correlate.set_defaults(handler=run_correlate)

    importance = subcommands.add_parser(
        "importance", parents=[common], help="Rank features against an outcome."
    )
    importance.add_argument(
        "--target", required=True, help="Outcome column to predict, e.g. Y2."
    )
    importance.add_argument(
        "--test-size",
        type=float,
        default=ImportanceConfig.test_size,
        help="Held-out fraction (default: %(default)s).",
    )
    importance.add_argument(
        "--permutation-repeats",
        type=int,
        default=ImportanceConfig.n_permutation_repeats,
        help="Shuffles per feature for permutation importance "
        "(default: %(default)s).",
    )
    importance.set_defaults(handler=run_importance)

    validate = subcommands.add_parser(
        "validate", parents=[common], help="Check an imputed dataset for defects."
    )
    validate.add_argument(
        "--original",
        type=Path,
        help="Source dataset, enabling the cross-frame checks.",
    )
    validate.add_argument(
        "--expect-skipped",
        nargs="*",
        default=[],
        metavar="COLUMN",
        help="Columns that were deliberately left unimputed.",
    )
    validate.set_defaults(handler=run_validate)

    return parser


def _load(args: argparse.Namespace, as_text: bool = False):
    """Read the input and classify its columns.

    Every subcommand starts here, so schema detection is guaranteed consistent
    across stages -- the exact problem the original scripts had.
    """
    logger.info("Reading %s", args.input)
    frame = source_for(args.input).read(as_text=as_text)
    classifier = ColumnClassifier(SchemaConfig(numeric_threshold=args.numeric_threshold))
    schema = classifier.classify(frame)
    logger.info("Loaded %s rows x %s columns (%s)", f"{len(frame):,}", frame.shape[1], schema.describe())
    return frame, schema, classifier


def _writer(args: argparse.Namespace) -> FigureWriter:
    """Build a figure writer for the run's output directory.

    FigureWriter creates the directory itself, so there is nothing to do here
    beyond installing the shared plot style.
    """
    plots = PlotConfig()
    apply_style(plots)
    return FigureWriter(output_dir=args.output_dir, config=plots)


def run_profile(args: argparse.Namespace) -> int:
    frame, schema, classifier = _load(args)
    profile = DatasetProfiler().profile(classifier.coerce(frame, schema), schema)
    report = profile.to_text()
    print(report)

    if args.save:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "profile.txt"
        path.write_text(report, encoding="utf-8")
        logger.info("Wrote %s", path)
    return 0


def run_impute(args: argparse.Namespace) -> int:
    # as_text=True: let the classifier decide types rather than pandas, so an
    # ID-like string is not silently reinterpreted before we see it.
    frame, schema, classifier = _load(args, as_text=True)

    config = ImputationConfig(
        n_neighbors=args.neighbors,
        skip_columns=tuple(args.skip),
        keep_original_columns=args.keep_originals,
    )
    imputed, report = ImputationPipeline(config, classifier).run(frame, schema)

    print(report.summary())
    if report.neighbors_used:
        print(f"\nKNN used k={report.neighbors_used}.")

    path = write_frame(imputed, args.output)
    logger.info("Wrote %s", path)
    return 0


def run_eda(args: argparse.Namespace) -> int:
    frame, schema, classifier = _load(args)
    coerced = classifier.coerce(frame, schema)

    analysis = ExploratoryAnalysis(_writer(args), EDAConfig())
    if args.skip_tsne:
        produced = [
            label
            for label, stage in (
                ("missingness", analysis.plot_missingness),
                ("distributions", analysis.plot_distributions),
                ("cluster map", analysis.plot_cluster_map),
                ("PCA", analysis.plot_pca),
                ("pair plot", analysis.plot_pairs),
            )
            if stage(coerced, schema)
        ]
    else:
        produced = analysis.run_all(coerced, schema)

    print(f"Generated {len(produced)} figure set(s) in {args.output_dir}: {', '.join(produced)}")
    return 0


def run_correlate(args: argparse.Namespace) -> int:
    frame, schema, classifier = _load(args)
    coerced = classifier.coerce(frame, schema)

    analyzer = CorrelationAnalyzer()
    result = analyzer.analyze(coerced, schema, method=args.method)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.matrix.to_csv(args.output_dir / f"correlation_matrix_{args.method}.csv")
    result.ranked_pairs.to_csv(
        args.output_dir / f"correlation_ranked_{args.method}.csv", index=False
    )
    analyzer.plot_heatmap(result, _writer(args))

    print(f"Top 10 {args.method} correlations:\n{result.top(10).to_string(index=False)}")
    strong = result.above(args.threshold)
    print(f"\n{len(strong)} pair(s) at or above |r| = {args.threshold}.")
    logger.info("Wrote tables and heatmap to %s", args.output_dir)
    return 0


def run_importance(args: argparse.Namespace) -> int:
    frame, schema, classifier = _load(args)
    coerced = classifier.coerce(frame, schema)

    config = ImportanceConfig(
        target_column=args.target,
        test_size=args.test_size,
        n_permutation_repeats=args.permutation_repeats,
    )
    study = FeatureImportanceStudy(config, _writer(args))
    result = study.run(coerced, schema)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"feature_importance_{args.target}.csv"
    result.rankings.to_csv(path)

    print(result.to_text())
    logger.info("Wrote %s", path)
    return 0


def run_validate(args: argparse.Namespace) -> int:
    imputed = source_for(args.input).read(as_text=True)
    original = source_for(args.original).read(as_text=True) if args.original else None

    validator = ImputationValidator(expected_skipped=tuple(args.expect_skipped))
    result = validator.validate(imputed, original)

    print(result.to_text())
    # Non-zero exit on failure so this can gate a CI step or a shell pipeline.
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns the process exit code."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
        stream=sys.stderr,
    )

    try:
        return args.handler(args)
    except DataSourceError as exc:
        logger.error("%s", exc)
        return 2
    except (ValueError, KeyError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
