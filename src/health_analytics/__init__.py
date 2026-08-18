"""Analysis pipeline for a clinical encounter dataset.

Five stages, each usable on its own:

* :mod:`~health_analytics.imputation` -- fill missing values
* :mod:`~health_analytics.profiling`  -- text summary of an unfamiliar table
* :mod:`~health_analytics.eda`        -- exploratory figures
* :mod:`~health_analytics.correlation`-- pairwise relationships
* :mod:`~health_analytics.importance` -- supervised feature ranking

Typical use::

    from pathlib import Path
    from health_analytics import ColumnClassifier, ImputationPipeline, source_for

    frame = source_for(Path("data/encounters.csv")).read(as_text=True)
    schema = ColumnClassifier().classify(frame)
    imputed, report = ImputationPipeline().run(frame, schema)
    print(report.summary())

Or from the command line: ``health-analytics impute --input data/encounters.csv``
"""

from __future__ import annotations

from .config import (
    EDAConfig,
    ImportanceConfig,
    ImputationConfig,
    PlotConfig,
    SchemaConfig,
)
from .correlation import CorrelationAnalyzer, CorrelationResult
from .datasets import CsvSource, DataSource, DataSourceError, ExcelSource, source_for, write_frame
from .eda import ExploratoryAnalysis
from .importance import (
    FeatureImportanceStudy,
    ImportanceStudyResult,
    ModelEvaluation,
    TaskInference,
    TaskType,
)
from .imputation import (
    ImputationPipeline,
    ImputationReport,
    KNNNumericImputer,
    ModeCategoricalImputer,
)
from .profiling import DatasetProfile, DatasetProfiler
from .schema import ColumnClassifier, DatasetSchema
from .validation import ImputationValidator, ValidationResult
from .visualization import FigureWriter, apply_style

__version__ = "1.0.0"

__all__ = [
    # Configuration
    "SchemaConfig",
    "ImputationConfig",
    "EDAConfig",
    "ImportanceConfig",
    "PlotConfig",
    # Loading
    "DataSource",
    "CsvSource",
    "ExcelSource",
    "DataSourceError",
    "source_for",
    "write_frame",
    # Schema
    "ColumnClassifier",
    "DatasetSchema",
    # Stages
    "ImputationPipeline",
    "ImputationReport",
    "KNNNumericImputer",
    "ModeCategoricalImputer",
    "DatasetProfiler",
    "DatasetProfile",
    "ExploratoryAnalysis",
    "CorrelationAnalyzer",
    "CorrelationResult",
    "FeatureImportanceStudy",
    "ImportanceStudyResult",
    "ModelEvaluation",
    "TaskType",
    "TaskInference",
    "ImputationValidator",
    "ValidationResult",
    # Plotting
    "FigureWriter",
    "apply_style",
    "__version__",
]
