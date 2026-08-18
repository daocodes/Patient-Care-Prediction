"""Post-imputation validation.

Imputation is easy to get subtly wrong in ways that leave a plausible-looking
file behind: a column silently skipped, rows reordered against the source, a
categorical field filled with the placeholder everywhere because its mode was
never computed. These checks run against the *output* file and fail loudly, so
a broken run is caught before anything is modelled on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .schema import ColumnClassifier


@dataclass
class ValidationIssue:
    """One failed check."""

    check: str
    column: str | None
    detail: str

    def __str__(self) -> str:
        location = f" [{self.column}]" if self.column else ""
        return f"{self.check}{location}: {self.detail}"


@dataclass
class ValidationResult:
    """Outcome of validating an imputed dataset."""

    issues: list[ValidationIssue] = field(default_factory=list)
    checks_run: int = 0

    @property
    def passed(self) -> bool:
        return not self.issues

    def add(self, check: str, detail: str, column: str | None = None) -> None:
        self.issues.append(ValidationIssue(check, column, detail))

    def to_text(self) -> str:
        if self.passed:
            return f"All {self.checks_run} validation checks passed."
        lines = [f"{len(self.issues)} issue(s) across {self.checks_run} checks:"]
        lines.extend(f"  - {issue}" for issue in self.issues)
        return "\n".join(lines)


class ImputationValidator:
    """Checks an imputed dataset against the source it was derived from."""

    def __init__(
        self,
        expected_skipped: tuple[str, ...] = (),
        classifier: ColumnClassifier | None = None,
    ) -> None:
        #: Columns intentionally left unimputed; exempt from the completeness
        #: check and required to be byte-identical to the source.
        self._expected_skipped = set(expected_skipped)
        self._classifier = classifier or ColumnClassifier()

    def validate(
        self, imputed: pd.DataFrame, original: pd.DataFrame | None = None
    ) -> ValidationResult:
        """Run every applicable check.

        Args:
            imputed: The imputation output.
            original: The source data. When supplied, enables the cross-frame
                checks (shape, column set, skipped columns unchanged) that
                catch the most damaging class of error.
        """
        result = ValidationResult()

        self._check_completeness(imputed, result)
        self._check_placeholder_saturation(imputed, result)
        self._check_no_constant_columns(imputed, result)

        if original is not None:
            self._check_shape_preserved(imputed, original, result)
            self._check_columns_preserved(imputed, original, result)
            self._check_skipped_unchanged(imputed, original, result)

        return result

    def _check_completeness(self, imputed: pd.DataFrame, result: ValidationResult) -> None:
        """No cell outside the skip list may still be blank.

        Counts empty strings as missing too. A whitespace-only cell reads as
        present to ``isna()`` but breaks the moment anything tries to parse it.
        """
        result.checks_run += 1
        for column in imputed.columns:
            if column in self._expected_skipped or column.endswith("_orig"):
                continue
            null_count = int(imputed[column].isna().sum())
            blank_count = int(
                (imputed[column].astype("string").str.strip() == "").sum()
            )
            if null_count or blank_count:
                result.add(
                    "incomplete",
                    f"{null_count} null and {blank_count} blank values remain",
                    column,
                )

    def _check_placeholder_saturation(
        self, imputed: pd.DataFrame, result: ValidationResult, threshold: float = 0.5
    ) -> None:
        """Flag columns that are mostly the MISSING placeholder.

        A column filled with the placeholder past this share was effectively
        empty in the source. It survives as a column but carries no signal, and
        silently joining it into a model is worse than dropping it knowingly.
        """
        result.checks_run += 1
        for column in imputed.columns:
            # Tested by "is it numeric?" rather than "is its dtype object?":
            # pandas 3.0 gives text columns a dedicated `str` dtype, so the
            # object comparison silently skipped every column it should check.
            if pd.api.types.is_numeric_dtype(imputed[column]):
                continue
            share = float((imputed[column] == "MISSING").mean())
            if share >= threshold:
                result.add(
                    "placeholder-saturated",
                    f"{share:.1%} of values are the MISSING placeholder",
                    column,
                )

    def _check_no_constant_columns(
        self, imputed: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Flag columns that ended up with a single distinct value.

        Usually means over-aggressive filling collapsed a sparse column onto
        its mode.
        """
        result.checks_run += 1
        for column in imputed.columns:
            if column.endswith("_orig"):
                continue
            if imputed[column].nunique(dropna=True) <= 1:
                result.add("constant", "column has a single distinct value", column)

    @staticmethod
    def _check_shape_preserved(
        imputed: pd.DataFrame, original: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Row count must be unchanged: imputation fills cells, never rows."""
        result.checks_run += 1
        if len(imputed) != len(original):
            result.add(
                "row-count",
                f"expected {len(original):,} rows, found {len(imputed):,}",
            )

    @staticmethod
    def _check_columns_preserved(
        imputed: pd.DataFrame, original: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Every source column must survive into the output."""
        result.checks_run += 1
        missing = set(original.columns) - set(imputed.columns)
        if missing:
            result.add("missing-columns", f"dropped: {', '.join(sorted(missing))}")

    def _check_skipped_unchanged(
        self, imputed: pd.DataFrame, original: pd.DataFrame, result: ValidationResult
    ) -> None:
        """Skipped columns must be identical to the source, cell for cell."""
        result.checks_run += 1
        for column in sorted(self._expected_skipped):
            if column not in imputed.columns or column not in original.columns:
                continue
            if not imputed[column].equals(original[column]):
                result.add("skipped-modified", "column was modified despite being skipped", column)
