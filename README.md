# health-analytics

A pipeline for a clinical encounter dataset: fill in missing values, profile
what you were given, look for structure, and find out which fields actually
predict an outcome.

The analysis this came out of placed **2nd at the International Big Data Health
Science Competition**, in the graduate division — I competed as an
undergraduate.

It began as five scripts I wrote while working through 100,000 patient
encounters. Each hardcoded its input path, and four carried their own copy of
the same "is this column numeric?" helper, drifted to three different
thresholds — so a column could be numeric in one script and categorical in the
next. This is the rewrite: one package, one implementation of each decision, and
96 tests.

## The data is not here

**The dataset is confidential patient data and is not in this repository.**
Nothing under `data/` is tracked, and `.gitignore` blocks data by extension
rather than by filename, so a new export is ignored by default.

So the pipeline stays runnable, `scripts/generate_sample_data.py` produces
synthetic data with the same shape and the same awkward parts: mixed types, ~3%
missingness, a column that is 72% empty, and free text inside otherwise numeric
fields. Every number below comes from that generator. The relationships in it
are planted — `X_signal` drives the outcomes, `X30`/`X31` are near-duplicates —
so the tests can assert the pipeline finds them.

## Running it

```bash
pip install -e ".[dev]"
python scripts/generate_sample_data.py --rows 5000 --output data/sample.csv
```

```bash
health-analytics profile    --input data/sample.csv
health-analytics impute     --input data/sample.csv --output data/imputed.csv --skip X60_sparse
health-analytics validate   --input data/imputed.csv --original data/sample.csv --expect-skipped X60_sparse
health-analytics eda        --input data/sample.csv --output-dir output/
health-analytics correlate  --input data/sample.csv --method spearman
health-analytics importance --input data/sample.csv --target Y2
```

## Results

`profile` answers the four questions worth asking of an unfamiliar table — what
is missing, what is redundant, what is skewed, what is effectively constant:

```
Dataset: 5,000 rows x 43 columns
Schema:  34 numeric, 9 categorical, 1 identifier, 5 outcome

MISSING VALUES (35 columns affected)
  X60_sparse            72.42%
  X24_oxygen_sat         4.28%

STRONGEST CORRELATIONS
  Y2                 X_signal            -0.980
  X30                X31                 +0.966

SKEWNESS (|skew| >= 1.0, consider a transform)
  X40_skewed              +4.88

CATEGORICAL DOMINANCE (one value covers >= 90%)
  X56_reminder         'email'   94.1%
```

It finds the planted duplicate pair, the skewed column, and the categorical that
is 94% one value, without being told what to look for.

`importance` trains a model on one outcome and ranks features three ways, since
each is wrong in its own direction: gain favours high-cardinality columns,
permutation splits credit between correlated features, SHAP is slow. A feature
ranking highly under all three is real signal.

```
Target: Y2
regression | train=4,000 test=1,000
  rmse=0.0680  mae=0.0538  r2=0.9514

                  gain  permutation
X_signal      1.341694     1.905281
X30           0.006859     0.000351
X60_sparse    0.005850     0.000342
T             0.001734     0.000300
```

The planted driver comes out on top under both methods, three orders of
magnitude clear of second place. Pointing the same command at `Y5`, which the
generator fills with random integers, gives 31% accuracy across three classes —
chance. That is the correct answer, and worth more than a good-looking number.

`impute` fills gaps with KNN for numbers and mode for labels; `validate` then
checks the output for the ways imputation actually goes wrong — rows dropped,
columns silently modified, a field left saturated with the placeholder.

Results from the real dataset are withheld, since they derive from confidential
records.

## Fixes the rewrite turned up

- **Gain importance was reporting zero for every feature.** XGBoost returns
  positional booster keys (`f0`, `f1`) when feature names are not retained; the
  original looked them up by column name, matched nothing, and defaulted to
  `0.0`. The committed output CSV has 51 rows and not one non-zero value.
  Nothing raised.
- **Permutation importance was scored on data the model never saw the shape
  of** — NaN replaced with `-999` after training with NaN intact.
- **The t-SNE subsample was unseeded**, so the projection changed every run, and
  the cluster map came out untitled because `plt.title()` after
  `sns.clustermap()` targets the colour bar's axes.

## Layout

```
src/health_analytics/
  schema.py       decide what each column is — everything defers to this
  datasets.py     read CSV or Excel behind one interface
  imputation.py   KNN for numbers, mode for labels, fallbacks for both
  profiling.py    text summary of an unfamiliar table
  eda.py          exploratory figures
  correlation.py  pairwise relationships, ranked
  importance.py   train a model, rank features three ways
  validation.py   check an imputed file for defects
  cli.py          six subcommands
```

```bash
pytest              # 96 tests, all synthetic fixtures
ruff check src tests
```

## Known limits

- Imputation is fit on the full dataset. Fine for producing an analysis file; if
  the output fed a model whose generalisation error mattered, the imputer would
  need to be fit on the training split alone.
- KNN imputation is O(n²) in rows — comfortable at 100k, not beyond.
- `--skip` is the only handling for a column that is 72% empty. Whether it
  should be imputed, dropped, or replaced by a missingness indicator is a
  modelling question this does not answer.
