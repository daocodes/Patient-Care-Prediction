"""Supervised feature importance.

Trains a gradient-boosted model on a chosen outcome and ranks the features
three different ways. Three, because each is wrong in its own direction:

* **Gain** is free (it falls out of training) but biased toward
  high-cardinality columns, which get more opportunities to split.
* **Permutation** measures the actual drop in held-out score when a column is
  shuffled, so it reflects predictive value -- but it splits credit
  unpredictably between correlated features.
* **SHAP** attributes each individual prediction and is the most faithful, but
  is also the slowest and an optional dependency.

A feature that ranks highly under all three is a real signal. One that ranks
highly under only gain usually is not.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from .config import ImportanceConfig
from .schema import DatasetSchema
from .visualization import FigureWriter, sns

logger = logging.getLogger(__name__)

# SHAP pulls in numba, which pins hard against the installed NumPy and is a
# frequent source of import failures. Importance analysis is useful without it,
# so the dependency is optional and its absence is reported, not fatal.
SHAP_IMPORT_ERROR = ""
try:
    import shap

    SHAP_AVAILABLE = True
except Exception as _shap_error:  # noqa: BLE001 - numba raises many error types
    shap = None  # type: ignore[assignment]
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = str(_shap_error)


class TaskType(str, Enum):
    """Whether the target calls for a classifier or a regressor."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class TaskInference:
    """Decides the task type from the target column's values.

    The rule: a target is classification if it is non-numeric, or if it is
    numeric with integer-only values and few distinct ones. Everything else is
    regression. This matters because the outcome columns here are a mix -- some
    are continuous scores, others are small integer codes -- and treating a
    5-level code as a regression target produces meaningless fractional
    predictions.
    """

    def __init__(self, max_classes: int = 10) -> None:
        self._max_classes = max_classes

    def infer(self, target: pd.Series) -> TaskType:
        numeric = pd.to_numeric(target, errors="coerce")
        observed = numeric.dropna()

        # Mostly-unparseable means the target holds labels, not measurements.
        if len(observed) / max(1, len(target)) <= 0.9:
            return TaskType.CLASSIFICATION
        if observed.empty:
            return TaskType.CLASSIFICATION

        values = observed.to_numpy(dtype=float)
        integral = np.all(np.isclose(np.modf(values)[0], 0.0))
        if integral and observed.nunique() <= self._max_classes:
            return TaskType.CLASSIFICATION
        return TaskType.REGRESSION

    @staticmethod
    def prepare_target(target: pd.Series, task: TaskType) -> pd.Series:
        """Cast the target to the dtype its task requires."""
        if task is TaskType.REGRESSION:
            return pd.to_numeric(target, errors="coerce").astype(float)
        numeric = pd.to_numeric(target, errors="coerce")
        if numeric.notna().all():
            return numeric.astype("int64")
        return target.astype(str)


class CategoricalEncoder:
    """Ordinal-encodes categorical columns while preserving missingness.

    XGBoost handles NaN natively and learns a default branch direction for it,
    so missing values are deliberately *not* filled here. Encoding them as a
    real category instead -- which is what happens if you pass a sentinel
    string straight through an encoder -- tells the model that "unknown" is a
    value patients can have, and it will happily split on it.
    """

    _SENTINEL = "__MISSING__"

    def fit_transform(self, features: pd.DataFrame, categorical: list[str]) -> pd.DataFrame:
        if not categorical:
            return features.copy()

        raw = features[categorical].astype(object)
        missing = raw.isna()

        # The encoder needs a hashable value in every cell; the sentinel is
        # swapped in for fitting and the NaNs are put back immediately after.
        filled = raw.where(~missing, self._SENTINEL)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded = pd.DataFrame(
            encoder.fit_transform(filled),
            columns=categorical,
            index=features.index,
        )
        encoded[missing] = np.nan

        numeric = features.drop(columns=categorical)
        # Reindex to the original column order so importance tables line up
        # with the source data.
        return pd.concat([numeric, encoded], axis=1)[features.columns]


@dataclass(frozen=True)
class ModelEvaluation:
    """Held-out scores for a trained model."""

    task: TaskType
    metrics: dict[str, float]
    n_train: int
    n_test: int

    def to_text(self) -> str:
        scores = "  ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return (
            f"{self.task.value} | train={self.n_train:,} test={self.n_test:,}\n"
            f"  {scores}"
        )


class ImportanceEstimator(ABC):
    """Ranks features by importance under one methodology."""

    name: str = "unknown"

    @abstractmethod
    def estimate(
        self, model, features: pd.DataFrame, target: pd.Series
    ) -> pd.Series:
        """Return importance per feature, indexed by feature name.

        Higher is more important. Implementations may return fewer entries than
        there are columns -- gain omits features the model never split on.
        """

    def plot(self, scores: pd.Series, writer: FigureWriter, top_n: int = 30) -> None:
        """Draw a horizontal bar chart of the top-``n`` features."""
        top = scores.sort_values(ascending=False).head(top_n)
        if top.empty:
            return
        # Grow the figure with the bar count so labels never overlap.
        height = max(4.0, 0.28 * len(top))
        with writer.figure(f"importance_{self.name}.png", size=(9, height)) as ax:
            sns.barplot(x=top.to_numpy(), y=list(top.index), ax=ax, color="steelblue")
            ax.set_xlabel(f"{self.name} importance")
            ax.set_title(f"Feature importance ({self.name})")


class GainImportance(ImportanceEstimator):
    """Total loss reduction contributed by each feature's splits."""

    name = "gain"

    def estimate(self, model, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        booster = model.get_booster()
        raw = booster.get_score(importance_type="gain")
        scores = self._resolve_names(raw, list(features.columns), booster.feature_names)
        # Features the model never split on are absent from the booster's dict;
        # fill them in at zero so every estimator returns the same index and
        # the three rankings can be joined into one table.
        return pd.Series(
            {column: scores.get(column, 0.0) for column in features.columns},
            name=self.name,
        )

    @staticmethod
    def _resolve_names(
        raw: dict[str, float], columns: list[str], booster_names: list[str] | None
    ) -> dict[str, float]:
        """Map booster keys back to column names.

        XGBoost only keeps feature names on the booster in some configurations;
        trained through the scikit-learn wrapper on a DataFrame it does not, and
        ``get_score`` returns positional keys -- ``f0``, ``f1`` -- instead. Looking
        those up by column name matches nothing and yields an importance of zero
        for every feature, which is what the original script reported for all 51
        of them without any error being raised.
        """
        if booster_names:
            return raw

        resolved: dict[str, float] = {}
        for key, value in raw.items():
            if key.startswith("f") and key[1:].isdigit():
                index = int(key[1:])
                if index < len(columns):
                    resolved[columns[index]] = value
            else:
                resolved[key] = value
        return resolved


class PermutationImportance(ImportanceEstimator):
    """Mean drop in held-out score when a feature's values are shuffled."""

    name = "permutation"

    def __init__(self, n_repeats: int = 10, random_state: int = 42) -> None:
        self._n_repeats = n_repeats
        self._random_state = random_state

    def estimate(self, model, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        # Passed through with NaN intact. The original filled missing values
        # with -999 first, which meant the model was scored on a distribution it
        # was never trained on -- the measured "importance" of a column with
        # many gaps was largely the cost of that substitution.
        result = permutation_importance(
            model,
            features,
            target,
            n_repeats=self._n_repeats,
            random_state=self._random_state,
            n_jobs=-1,
        )
        return pd.Series(
            result.importances_mean, index=features.columns, name=self.name
        )


class ShapImportance(ImportanceEstimator):
    """Mean absolute SHAP value per feature.

    Requires the optional ``shap`` extra. :meth:`available` lets callers skip
    it cleanly rather than catching an ImportError at the call site.
    """

    name = "shap"

    def __init__(self, max_rows: int = 2_000, random_state: int = 42) -> None:
        self._max_rows = max_rows
        self._random_state = random_state
        self._values = None
        self._sample: pd.DataFrame | None = None

    @staticmethod
    def available() -> bool:
        return SHAP_AVAILABLE

    def estimate(self, model, features: pd.DataFrame, target: pd.Series) -> pd.Series:
        if not SHAP_AVAILABLE:
            raise RuntimeError(f"shap is not importable: {SHAP_IMPORT_ERROR}")

        # SHAP is exact for trees but scales with rows x features x trees;
        # sampling keeps a 100k-row test set to a tractable runtime.
        sample = features
        if len(sample) > self._max_rows:
            sample = sample.sample(self._max_rows, random_state=self._random_state)

        values = shap.TreeExplainer(model).shap_values(sample)
        self._values, self._sample = values, sample
        return pd.Series(self._mean_abs(values), index=sample.columns, name=self.name)

    @staticmethod
    def _mean_abs(values) -> np.ndarray:
        """Collapse a SHAP result to one score per feature.

        The output shape depends on the task and the shap version. Binary and
        regression give ``(rows, features)``; multiclass gives either a list of
        those, one per class, or a single ``(rows, features, classes)`` array.
        All three collapse to a per-feature mean absolute attribution.
        """
        if isinstance(values, list):
            # (classes, rows, features) -> mean over classes and rows.
            return np.stack([np.abs(v) for v in values]).mean(axis=(0, 1))
        array = np.abs(np.asarray(values))
        if array.ndim == 3:
            array = array.mean(axis=2)
        return array.mean(axis=0)

    def plot_summary(self, writer: FigureWriter) -> None:
        """Save SHAP's beeswarm summary, if :meth:`estimate` has been called."""
        if self._values is None or self._sample is None:
            return
        import matplotlib.pyplot as plt

        shap.summary_plot(self._values, self._sample, show=False)
        writer.save_existing(plt.gcf(), "shap_summary.png")


@dataclass(frozen=True)
class ImportanceStudyResult:
    """Everything one importance run produced."""

    evaluation: ModelEvaluation
    rankings: pd.DataFrame
    target_column: str

    def top(self, n: int = 15, by: str = "permutation") -> pd.DataFrame:
        """Top ``n`` features by one estimator's column."""
        column = by if by in self.rankings.columns else self.rankings.columns[0]
        return self.rankings.sort_values(column, ascending=False).head(n)

    def to_text(self, n: int = 15) -> str:
        return (
            f"Target: {self.target_column}\n"
            f"{self.evaluation.to_text()}\n\n"
            f"Top {n} features:\n{self.top(n).to_string()}"
        )


class FeatureImportanceStudy:
    """Trains a model on one outcome and ranks its features.

    Sequences the whole flow -- infer the task, encode, split, fit, score, then
    run each estimator -- and returns a single table with one row per feature
    and one column per methodology, so the three rankings can be compared
    directly instead of living in three separate files.
    """

    def __init__(
        self,
        config: ImportanceConfig,
        writer: FigureWriter | None = None,
    ) -> None:
        self._config = config
        self._writer = writer
        self._task_inference = TaskInference(config.max_classification_classes)
        self._encoder = CategoricalEncoder()
        self._label_encoder: LabelEncoder | None = None

    def run(self, frame: pd.DataFrame, schema: DatasetSchema) -> ImportanceStudyResult:
        """Execute the full study against ``config.target_column``.

        Raises:
            ValueError: If the target is absent, or no rows have an observed
                target value.
        """
        target_column = self._config.target_column
        if target_column not in frame.columns:
            raise ValueError(
                f"Target '{target_column}' not in dataset. "
                f"Available outcomes: {', '.join(schema.outcomes) or 'none detected'}"
            )

        features, target = self._prepare(frame, schema, target_column)
        task = self._task_inference.infer(target)
        target = self._task_inference.prepare_target(target, task)
        logger.info("Inferred task: %s", task.value)

        if task is TaskType.CLASSIFICATION:
            # XGBoost requires class labels to be 0..n-1 contiguous integers;
            # the raw codes in this data are neither.
            self._label_encoder = LabelEncoder()
            target = pd.Series(
                self._label_encoder.fit_transform(target), index=target.index
            )

        X_train, X_test, y_train, y_test = self._split(features, target, task)
        model = self._fit(X_train, y_train, task)
        evaluation = self._evaluate(model, X_test, y_test, task, len(X_train))
        rankings = self._rank(model, X_test, y_test, features)

        return ImportanceStudyResult(
            evaluation=evaluation, rankings=rankings, target_column=target_column
        )

    def _prepare(
        self, frame: pd.DataFrame, schema: DatasetSchema, target_column: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Build the feature matrix and target, dropping unusable rows.

        Rows with no target value are removed: there is nothing to learn from
        them, and they cannot contribute to a held-out score either. Other
        outcome columns are excluded from the features -- Y1..Y5 describe the
        same encounters, so leaving them in leaks the answer.
        """
        target_raw = frame[target_column]
        observed = target_raw.notna()
        dropped = int((~observed).sum())
        if dropped:
            logger.info("Dropped %d rows with a missing target", dropped)
        if not observed.any():
            raise ValueError(f"No rows have an observed value for '{target_column}'")

        usable = frame.loc[observed]
        feature_columns = [
            c for c in schema.features(exclude=(target_column,)) if c in usable.columns
        ]
        features = usable[feature_columns]

        categorical = [c for c in schema.categorical if c in feature_columns]
        encoded = self._encoder.fit_transform(features, categorical)
        # Anything still non-numeric after encoding (a stray date string) would
        # be rejected by XGBoost, so coerce and let it become NaN.
        encoded = encoded.apply(pd.to_numeric, errors="coerce")

        return encoded, usable[target_column]

    def _split(self, features: pd.DataFrame, target: pd.Series, task: TaskType):
        """Hold out a test set, stratifying when the task allows it.

        Stratification keeps rare classes represented in both halves, but fails
        on any class with a single member, so it is applied only when every
        class has at least two.
        """
        stratify = None
        if task is TaskType.CLASSIFICATION:
            counts = target.value_counts()
            if len(counts) > 1 and counts.min() >= 2:
                stratify = target

        return train_test_split(
            features,
            target,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
            stratify=stratify,
        )

    def _fit(self, X_train: pd.DataFrame, y_train: pd.Series, task: TaskType):
        """Train the model. NaN is passed through, not filled.

        XGBoost learns a default direction for missing values at each split,
        which beats imputing a value the column never actually takes.
        """
        import xgboost as xgb

        common = {
            "random_state": self._config.random_state,
            "n_jobs": -1,
            "tree_method": "hist",
        }
        if task is TaskType.CLASSIFICATION:
            n_classes = int(y_train.nunique())
            model = xgb.XGBClassifier(
                eval_metric="logloss" if n_classes == 2 else "mlogloss",
                objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                **common,
            )
        else:
            model = xgb.XGBRegressor(eval_metric="rmse", **common)

        logger.info("Fitting %s on %d rows...", type(model).__name__, len(X_train))
        model.fit(X_train, y_train)
        return model

    def _evaluate(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        task: TaskType,
        n_train: int,
    ) -> ModelEvaluation:
        """Score the model on held-out data.

        Reports more than one metric on purpose: accuracy alone is misleading
        on the imbalanced outcomes here, where predicting the majority class
        for every patient already scores well.
        """
        predictions = model.predict(X_test)
        metrics: dict[str, float] = {}

        if task is TaskType.CLASSIFICATION:
            metrics["accuracy"] = float(accuracy_score(y_test, predictions))
            metrics["f1_macro"] = float(
                f1_score(y_test, predictions, average="macro", zero_division=0)
            )
            if y_test.nunique() == 2:
                probabilities = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, probabilities))
        else:
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, predictions)))
            metrics["mae"] = float(mean_absolute_error(y_test, predictions))
            metrics["r2"] = float(r2_score(y_test, predictions))

        return ModelEvaluation(
            task=task, metrics=metrics, n_train=n_train, n_test=len(X_test)
        )

    def _rank(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run every available estimator and join the results into one table.

        An estimator that fails is logged and omitted rather than aborting the
        study, which matters most for SHAP: it is the one with a fragile
        dependency chain and the one most likely to be unavailable.
        """
        estimators: list[ImportanceEstimator] = [
            GainImportance(),
            PermutationImportance(
                n_repeats=self._config.n_permutation_repeats,
                random_state=self._config.random_state,
            ),
        ]
        if ShapImportance.available():
            estimators.append(
                ShapImportance(
                    max_rows=self._config.max_shap_rows,
                    random_state=self._config.random_state,
                )
            )
        else:
            logger.warning("shap unavailable, skipping SHAP importance (%s)", SHAP_IMPORT_ERROR)

        columns: list[pd.Series] = []
        for estimator in estimators:
            try:
                logger.info("Computing %s importance...", estimator.name)
                scores = estimator.estimate(model, X_test, y_test)
                columns.append(scores)
                if self._writer is not None:
                    estimator.plot(scores, self._writer, self._config.top_n_features)
                    if isinstance(estimator, ShapImportance):
                        estimator.plot_summary(self._writer)
            except Exception:  # noqa: BLE001 - one estimator must not sink the run
                logger.exception("Skipping %s importance", estimator.name)

        if not columns:
            return pd.DataFrame(index=pd.Index(features.columns, name="feature"))

        rankings = pd.concat(columns, axis=1)
        rankings.index.name = "feature"
        # Sort by permutation where available: it is the only one of the three
        # measured against held-out predictive performance.
        sort_column = "permutation" if "permutation" in rankings.columns else rankings.columns[0]
        return rankings.sort_values(sort_column, ascending=False)
