# health-analytics

A pipeline for working with a clinical encounter dataset: fill in missing
values, profile what you were given, look for structure, and find out which
fields actually predict an outcome.

It started as five scripts I wrote while working through a 100,000-row dataset
of patient encounters. They worked, but each one hardcoded its input path at the
top, and four of them carried their own copy of the same "is this column
numeric?" helper. The copies had drifted to three different thresholds, so the
same column could be numeric in one script and categorical in the next. This is
the rewrite: one package, one implementation of each decision, and a test suite.

## About the data

**The dataset is confidential patient data and is not in this repository.**

Nothing under `data/` is tracked except its README, and `.gitignore` blocks the
relevant extensions by pattern rather than by filename, so a new export is
ignored by default instead of needing to be added to the ignore list first.

So that the pipeline is runnable by anyone, `scripts/generate_sample_data.py`
produces synthetic data with the same shape and the same awkward parts: mixed
column types, ~3% missingness, one column that is 72% empty, and free text
sitting inside otherwise numeric fields. Every output shown below came from
that generator, not from the real data.

The relationships in it are planted and known, which makes it more than a demo.
`X_signal` drives the outcomes, so a correct importance ranking has to surface
it. `X30` and `X31` are near-duplicates, so correlation analysis should catch
them. If a change breaks either, the tests fail.

## Running it

```bash
pip install -e ".[dev]"
python scripts/generate_sample_data.py --rows 5000 --output data/sample.csv
```

Then any of six subcommands:

```bash
health-analytics profile    --input data/sample.csv
health-analytics impute     --input data/sample.csv --output data/imputed.csv --skip X60_sparse
health-analytics validate   --input data/imputed.csv --original data/sample.csv --expect-skipped X60_sparse
health-analytics eda        --input data/sample.csv --output-dir output/
health-analytics correlate  --input data/sample.csv --method spearman
health-analytics importance --input data/sample.csv --target Y2
```

`profile` is the one to start with. It answers the four questions worth asking
of an unfamiliar table — what is missing, what is redundant, what is skewed,
and what is effectively constant:

```
Dataset: 5,000 rows x 43 columns
Schema:  34 numeric, 9 categorical, 1 identifier, 5 outcome

MISSING VALUES (35 columns affected)
  X60_sparse            72.42%
  X24_oxygen_sat         4.28%
  X23_temperature        4.18%

STRONGEST CORRELATIONS
  Y2                 X_signal            -0.980
  X30                X31                 +0.966
  Y1                 X_signal            +0.965

SKEWNESS (|skew| >= 1.0, consider a transform)
  X40_skewed              +4.88

CATEGORICAL DOMINANCE (one value covers >= 90%)
  X56_reminder         'email'   94.1%
```

`importance` trains a model on one outcome and ranks the features three ways:

```
Target: Y2
regression | train=4,000 test=1,000
  rmse=0.0680  mae=0.0538  r2=0.9514

Top 5 features:
                  gain  permutation
feature
X_signal      1.341694     1.905281
X30           0.006859     0.000351
X60_sparse    0.005850     0.000342
T             0.001734     0.000300
X3            0.003956     0.000266
```

The planted driver comes out on top under both methods, and the gap to second
place is three orders of magnitude. Pointing the same command at `Y5` — which
the generator fills with random integers — gives 31% accuracy across three
classes, which is chance. That is the intended answer, and it is worth more
than a number that looks good.

## How it is put together

Six modules, each doing one thing, with the sequencing kept out of them.

| Module | Responsibility |
|---|---|
| `schema.py` | Decide what each column *is*. Everything else defers to this. |
| `datasets.py` | Read CSV or Excel behind one interface, raise a typed error. |
| `imputation.py` | Fill gaps: KNN for numbers, mode for labels, fallbacks for both. |
| `profiling.py` | Text summary of an unfamiliar table. |
| `eda.py` | Exploratory figures. |
| `correlation.py` | Pairwise relationships, ranked. |
| `importance.py` | Train a model, rank features three ways. |
| `validation.py` | Check an imputed file for the ways imputation goes wrong. |

Three decisions are load-bearing:

**Column typing lives in exactly one place.** It was duplicated four times and
the copies disagreed, which is the bug that motivated the rewrite. A column now
counts as numeric when at least 90% of its non-null values parse as numbers,
because real clinical exports put `pending` and `<90` in fields that are
otherwise numeric, and an all-or-nothing test demotes those columns to
90,000-category strings.

**Configuration is frozen dataclasses, not module globals.** Every stage takes
its settings as an argument. That is what makes the stages testable without
touching a file, and it is why the CLI can expose `--neighbors` and
`--numeric-threshold` without any stage knowing the CLI exists.

**Estimators sit behind a common interface.** Gain, permutation, and SHAP each
answer "which features matter" and each is wrong in its own direction — gain
favours high-cardinality columns, permutation splits credit unpredictably
between correlated features, SHAP is slow and has a fragile dependency chain.
They return the same shape, so the three rankings join into one table and can be
compared. A feature that ranks highly under all three is a real signal; one that
ranks highly only under gain usually is not.

SHAP is an optional extra for a practical reason: it depends on numba, which
pins hard against NumPy. It does not import in my current environment, and the
importance stage reports the other two rankings rather than failing.

## What the rewrite fixed

Consolidating the scripts turned up real bugs, not just style problems.

**Gain importance was silently reporting zero for every feature.** XGBoost's
booster only keeps feature names in some configurations; trained through the
scikit-learn wrapper on a DataFrame it does not, and `get_score()` returns
positional keys (`f0`, `f1`) instead. The original looked those up by column
name, matched nothing, and defaulted to `0.0`. The committed output CSV has 51
rows and not one non-zero gain value. Nothing raised. There are now regression
tests covering both the name mapping and the end-to-end result.

**Permutation importance was scored on data the model had never seen the shape
of.** The model trains with NaN intact, since XGBoost learns a default branch
direction for missing values. The original then filled NaN with `-999` before
measuring permutation importance, so what it measured for any column with gaps
was largely the cost of that substitution. NaN is now passed through.

**The cluster map was untitled.** `plt.title()` after `sns.clustermap()` targets
whatever axes happen to be current, which is the colour bar. Figures are now
written through a context manager that owns the figure it saves and closes it on
the way out, including when the plotting code raises.

**The t-SNE sample was unseeded**, so the projection changed on every run.

**Pair deduplication was O(n²) in Python.** The original built a sorted tuple
per row with `DataFrame.apply` to drop mirrored pairs — about 4,900 calls for 70
columns. A NumPy triangular mask does it in one operation.

Smaller ones: an `IndexError` waiting on the first fully-empty categorical
column, bare `except:` clauses swallowing everything including `KeyboardInterrupt`,
and `use_label_encoder=False`, which was removed in XGBoost 2.0.

Two came from library versions moving underneath the code, and both were caught
by tests rather than by reading: `DataFrame.stack()` stopped dropping NaN in
pandas 3.0, which let the masked half of the correlation matrix through, and
text columns stopped having dtype `object`, which made a validation check skip
every column it was supposed to inspect.

## Tests

```bash
pytest          # 96 tests
ruff check src tests
```

Every fixture is synthetic. No test reads the real data, because the tests have
to run for anyone who clones this.

They are mostly behavioural rather than mechanical. Imputation is checked for
the properties that actually matter — that observed values are never overwritten,
that row order survives, that an integer column does not come back as
`3.0000000004`, and that standardisation stops a column measured in tens of
thousands from dominating the neighbour distance. Task inference is checked at
the boundary where a small-integer target stops being a class and becomes a
quantity.

## Known limits

- **Imputation is fit on the full dataset**, including rows that later land in a
  test split. For a pipeline whose output is an analysis file this is fine; if
  the imputed data fed a model whose generalisation error mattered, the imputer
  would need to be fit on the training split alone.
- **KNN imputation is O(n²) in rows.** It is comfortable at 100k rows on 45
  numeric columns but will not stay that way. Chunking or an approximate
  neighbour index is the next step.
- **Nothing is parallelised across stages**, and the stages are independent
  enough that they could be.
- **`--skip` is the only handling for a column that is 72% empty.** Deciding
  whether such a column should be imputed at all, dropped, or replaced by a
  missingness indicator is a modelling question the pipeline does not answer.
