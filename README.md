# Patient Care Prediction Model

A pipeline for taking messy patient data and turning it into insights through a predictive model. The model predicts if it is in the best interest for a patient to get surgery based on factors like salary, BMI, pre-existing conditions, and other important details.


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

```

`importance` trains a model on one outcome and ranks the features three ways:

```


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


## Tests

```bash
pytest          # 96 tests
ruff check src tests
```

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
