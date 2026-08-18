#!/usr/bin/env python3
"""Generate a synthetic dataset matching the reference schema.

The real data is confidential and is not distributed with this project, so
without this the pipeline could not be run by anyone who clones the repository.
The output mimics the shape and the awkward parts of a real clinical export --
mixed column types, ~3% missingness, one nearly-empty column, free text inside
otherwise numeric fields -- so every code path gets exercised.

The relationships are planted and known, which makes this useful beyond a demo:
`X_signal` drives the outcomes, so a correct importance ranking has to surface
it, and `X30`/`X31` are near-duplicates that correlation analysis should catch.

    python scripts/generate_sample_data.py --rows 5000 --output data/sample.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEPARTMENTS = ["GEN", "SURG", "PSY", "OBGYN"]
VISIT_TYPES = ["follow-up", "urgent_care", "new_patient"]
REFERRALS = ["PCP", "self", "specialist", "employer"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
WEATHER = ["sunny", "rain", "cloudy"]
CHANNELS = ["email", "text", "phone", "portal"]
PAYMENTS = ["card", "insurance", "cash"]
LANGUAGES = ["EN", "ES"]


def build(rows: int, seed: int) -> pd.DataFrame:
    """Assemble the synthetic frame."""
    rng = np.random.default_rng(seed)

    # The planted driver. Everything downstream is a function of this plus noise,
    # so any importance method worth using must rank it first.
    signal = rng.normal(size=rows)
    treatment = (signal + rng.normal(scale=0.5, size=rows) > 0).astype(int)

    frame = pd.DataFrame({"PAT_ID": np.arange(1, rows + 1)})

    # Outcomes: three continuous, one integer-coded, one count. The mix is
    # deliberate -- it exercises both branches of task inference.
    frame["Y1"] = 1 / (1 + np.exp(-signal)) + rng.normal(scale=0.05, size=rows)
    frame["Y2"] = 0.5 - 0.3 * signal + rng.normal(scale=0.06, size=rows)
    frame["Y3"] = 0.4 + 0.25 * signal + rng.normal(scale=0.08, size=rows)
    frame["Y4"] = np.clip(5 + 4 * signal + rng.normal(size=rows), 0, None)
    frame["Y5"] = rng.integers(0, 3, rows)  # small integer codes
    frame["T"] = treatment
    frame["X_signal"] = signal

    # Binary flags, mostly noise.
    for i in range(1, 13):
        frame[f"X{i}"] = rng.integers(0, 2, rows)

    # Vitals, on realistic scales. The wide spread across columns is the point:
    # it is what makes standardisation necessary before KNN imputation.
    frame["X20_heart_rate"] = rng.normal(78, 12, rows).round(0)
    frame["X21_systolic"] = rng.normal(122, 16, rows).round(0)
    frame["X22_diastolic"] = rng.normal(79, 10, rows).round(0)
    frame["X23_temperature"] = rng.normal(36.8, 0.4, rows).round(2)
    frame["X24_oxygen_sat"] = np.clip(rng.normal(97, 2, rows), 80, 100).round(0)
    frame["X25_weight_lb"] = rng.normal(172, 38, rows).round(1)
    frame["X26_bmi"] = rng.normal(27, 6, rows).round(2)
    frame["X27_cost"] = np.abs(rng.normal(24_000, 9_000, rows)).round(2)

    # A pair of near-duplicates, so correlation analysis has something real to
    # find. X31 is X30 plus a little noise.
    frame["X30"] = rng.normal(100, 15, rows).round(2)
    frame["X31"] = (frame["X30"] + rng.normal(scale=4, size=rows)).round(2)

    # Heavily skewed, to give the profiler's skew check something to report.
    frame["X40_skewed"] = rng.exponential(scale=3, size=rows).round(2) ** 2

    # Categoricals, one of them deliberately dominated by a single value.
    frame["X50_department"] = rng.choice(DEPARTMENTS, rows)
    frame["X51_visit_type"] = rng.choice(VISIT_TYPES, rows)
    frame["X52_referral"] = rng.choice(REFERRALS, rows)
    frame["X53_weekday"] = rng.choice(WEEKDAYS, rows)
    frame["X54_season"] = rng.choice(SEASONS, rows)
    frame["X55_weather"] = rng.choice(WEATHER, rows)
    frame["X56_reminder"] = rng.choice(CHANNELS, rows, p=[0.94, 0.03, 0.02, 0.01])
    frame["X57_payment"] = rng.choice(PAYMENTS, rows)
    frame["X58_language"] = rng.choice(LANGUAGES, rows, p=[0.85, 0.15])

    # Lowercase, matching the real export's inconsistent casing.
    frame["chk_hour"] = np.clip(rng.normal(11, 3, rows), 6, 20).round(3)
    frame["x13"] = rng.normal(23_600, 60, rows).round(4)

    _inject_missingness(frame, rng)
    _inject_free_text(frame, rng)
    return frame


def _inject_missingness(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """Punch holes at roughly the rate the real data has.

    Identifiers and outcomes are left intact; the sparse column is separate,
    because a column that is 73% empty is a different problem from one that is
    3% empty and the pipeline should be seen handling both.
    """
    protected = {"PAT_ID", "Y1", "Y2", "Y3", "Y4", "Y5", "T", "X_signal"}
    rows = len(frame)

    for column in frame.columns:
        if column in protected:
            continue
        mask = rng.random(rows) < 0.03
        frame.loc[mask, column] = np.nan

    # One near-empty column, mirroring X51 in the reference data.
    frame["X60_sparse"] = np.where(
        rng.random(rows) < 0.27, rng.normal(50, 10, rows).round(2), np.nan
    )


def _inject_free_text(frame: pd.DataFrame, rng: np.random.Generator) -> None:
    """Drop occasional text into numeric columns.

    Real clinical exports contain entries like "pending" and "<0.1" in fields
    that are otherwise numeric. This is the reason type detection uses a
    threshold instead of requiring every value to parse.
    """
    for column, token in (("X23_temperature", "pending"), ("X24_oxygen_sat", "<90")):
        mask = rng.random(len(frame)) < 0.01
        frame[column] = frame[column].astype(object)
        frame.loc[mask, column] = token


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5_000, help="Rows to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output", type=Path, default=Path("data/sample.csv"), help="Output path."
    )
    args = parser.parse_args()

    frame = build(args.rows, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)

    missing = frame.isna().to_numpy().sum()
    print(
        f"Wrote {args.output}: {len(frame):,} rows x {frame.shape[1]} columns, "
        f"{missing:,} missing cells ({missing / frame.size:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
