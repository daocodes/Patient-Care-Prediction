# Data directory

**This directory is intentionally empty, and must stay that way.**

The dataset this pipeline was built for is confidential patient data. It is not
in this repository, it is not in the commit history, and it must never be added
to either. `.gitignore` blocks the relevant file extensions by pattern rather
than by filename, so a newly created export is ignored by default instead of
having to be added to the ignore list first. Check `git status` before you
commit regardless.

Put your own copy here — anything in this directory except this file is
ignored — or point `--input` anywhere else on disk.

## Expected shape

The pipeline does not require these exact columns. Names are only used for the
two roles that cannot be inferred from values, and both are configurable in
[`SchemaConfig`](../src/health_analytics/config.py):

| Role | Default | Why it is named rather than inferred |
|---|---|---|
| Identifier | `PAT_ID` | A patient ID is perfectly numeric, so type detection would treat it as a feature. It also correlates with anything that drifted over the collection period, which is an artefact of row order. |
| Outcomes | `Y1`–`Y5` | Predicting one outcome with the others in the feature matrix leaks the answer. |

Everything else is classified by inspecting its values, so a dataset with
entirely different column names works without code changes.

## Reference schema

The dataset the pipeline was developed against had 100,000 rows and 70 columns:

| Group | Columns | Type | Notes |
|---|---:|---|---|
| Identifier | `PAT_ID` | integer | One row per encounter |
| Outcomes | `Y1`–`Y5` | mixed | `Y1`–`Y3` continuous scores, `Y4`–`Y5` continuous, some variants integer-coded |
| Treatment | `T` | binary | |
| Features | `X1`–`X62`, `x13`, `chk_hour` | mixed | Binary flags, vitals, categorical labels, and two datetime columns |

Feature columns cover roughly four kinds of field:

- **Binary flags** — `X1`–`X12`, and others
- **Vitals and measurements** — heart rate, systolic/diastolic pressure, temperature,
  oxygen saturation, weight, BMI
- **Categorical labels** — department, visit type, referral source, day of week,
  season, time of day, weather, reminder channel, payment method, language
- **Datetimes** — appointment and check-in timestamps

Note the inconsistent casing of `x13` against `X1`–`X62`. It is preserved as-is:
column names are treated as opaque, so a real export's quirks do not need
cleaning before the pipeline will run.

## Characteristics worth knowing

Two properties of the reference data shaped the design:

- **Missingness is around 3% per column**, spread across roughly 50 of the 70
  columns, with one column (`X51`) missing 73% of its values. A column that
  sparse is a candidate for `--skip` rather than imputation: filling it means
  inventing three quarters of it.
- **Numeric columns hold occasional free text.** Detection therefore treats a
  column as numeric when 90% of its non-null values parse as numbers, instead of
  requiring all of them. See `--numeric-threshold` to adjust.
