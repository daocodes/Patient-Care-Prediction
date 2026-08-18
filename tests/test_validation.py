"""Tests for post-imputation validation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from health_analytics.validation import ImputationValidator


class TestCompleteness:
    def test_a_complete_frame_passes(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "b": list("xyz")})
        assert ImputationValidator().validate(frame).passed

    def test_remaining_nulls_are_reported(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": list("xyz")})
        result = ImputationValidator().validate(frame)
        assert not result.passed
        assert any(i.check == "incomplete" and i.column == "a" for i in result.issues)

    def test_blank_strings_count_as_missing(self) -> None:
        """A whitespace cell reads as present but breaks the first parse."""
        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "  ", "z"]})
        result = ImputationValidator().validate(frame)
        assert any(i.check == "incomplete" and i.column == "b" for i in result.issues)

    def test_expected_skipped_columns_are_exempt(self) -> None:
        frame = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": list("xyz")})
        result = ImputationValidator(expected_skipped=("a",)).validate(frame)
        assert not any(i.check == "incomplete" for i in result.issues)


class TestQualityChecks:
    def test_placeholder_saturation_is_flagged(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["MISSING"] * 3 + ["real"]})
        result = ImputationValidator().validate(frame)
        assert any(i.check == "placeholder-saturated" for i in result.issues)

    def test_constant_columns_are_flagged(self) -> None:
        frame = pd.DataFrame({"a": [1, 2, 3], "flat": [9, 9, 9]})
        result = ImputationValidator().validate(frame)
        assert any(i.check == "constant" and i.column == "flat" for i in result.issues)


class TestCrossFrameChecks:
    def test_dropped_rows_are_caught(self) -> None:
        original = pd.DataFrame({"a": [1, 2, 3, 4]})
        imputed = pd.DataFrame({"a": [1, 2, 3]})
        result = ImputationValidator().validate(imputed, original)
        assert any(i.check == "row-count" for i in result.issues)

    def test_dropped_columns_are_caught(self) -> None:
        original = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        imputed = pd.DataFrame({"a": [1, 2]})
        result = ImputationValidator().validate(imputed, original)
        assert any(i.check == "missing-columns" for i in result.issues)

    def test_a_modified_skip_column_is_caught(self) -> None:
        """The point of skipping a column is that it comes back untouched."""
        original = pd.DataFrame({"keep": [1.0, np.nan, 3.0], "other": [1, 2, 3]})
        imputed = pd.DataFrame({"keep": [1.0, 2.0, 3.0], "other": [1, 2, 3]})
        result = ImputationValidator(expected_skipped=("keep",)).validate(imputed, original)
        assert any(i.check == "skipped-modified" for i in result.issues)

    def test_an_untouched_skip_column_passes(self) -> None:
        original = pd.DataFrame({"keep": [1.0, np.nan, 3.0], "other": [1, 2, 3]})
        result = ImputationValidator(expected_skipped=("keep",)).validate(
            original.copy(), original
        )
        assert not any(i.check == "skipped-modified" for i in result.issues)


class TestReporting:
    def test_issue_renders_with_its_column(self) -> None:
        result = ImputationValidator().validate(pd.DataFrame({"a": [1.0, np.nan]}))
        assert "[a]" in str(result.issues[0])

    def test_passing_summary_names_the_check_count(self) -> None:
        result = ImputationValidator().validate(pd.DataFrame({"a": [1, 2], "b": [3, 4]}))
        assert "checks passed" in result.to_text()
