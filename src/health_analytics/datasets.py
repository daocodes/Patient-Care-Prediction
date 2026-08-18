"""Dataset loading.

The original scripts each opened their input inline and reacted to failure with
``print(...); exit(1)``, which makes them impossible to call from anything else.
Loading lives here instead, raises a typed exception, and lets the CLI decide
how to present the error.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class DataSourceError(RuntimeError):
    """Raised when a dataset cannot be located, opened, or parsed."""


class DataSource(ABC):
    """A readable tabular source.

    Subclasses exist per file format. New formats (Parquet, a database query)
    only need to implement :meth:`read`; nothing downstream changes, because
    every stage takes a ``DataFrame``.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @abstractmethod
    def read(self, *, as_text: bool = False) -> pd.DataFrame:
        """Load the source into a DataFrame.

        Args:
            as_text: Read every cell as an object without letting the parser
                infer types. Required for the imputation stage, where pandas'
                inference would silently rewrite ID-like strings and dates
                before we get a chance to classify them ourselves.
        """

    def _check_exists(self) -> None:
        if not self.path.exists():
            raise DataSourceError(
                f"Input file not found: {self.path}\n"
                f"This project ships without data (see data/README.md). "
                f"Point --input at your own copy."
            )


class CsvSource(DataSource):
    """Reads delimited text files."""

    def __init__(self, path: Path | str, separator: str = ",", encoding: str = "utf-8") -> None:
        super().__init__(path)
        self.separator = separator
        self.encoding = encoding

    def read(self, *, as_text: bool = False) -> pd.DataFrame:
        self._check_exists()
        try:
            return pd.read_csv(
                self.path,
                sep=self.separator,
                encoding=self.encoding,
                # dtype=object disables inference; low_memory=False stops pandas
                # from choosing a different dtype per chunk on wide files.
                dtype=object if as_text else None,
                low_memory=False,
            )
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise DataSourceError(f"Failed to read CSV {self.path}: {exc}") from exc


class ExcelSource(DataSource):
    """Reads ``.xlsx``/``.xls`` workbooks."""

    def __init__(self, path: Path | str, sheet_name: str | int = 0) -> None:
        super().__init__(path)
        self.sheet_name = sheet_name

    def read(self, *, as_text: bool = False) -> pd.DataFrame:
        self._check_exists()
        try:
            return pd.read_excel(
                self.path,
                sheet_name=self.sheet_name,
                dtype=object if as_text else None,
            )
        except (ValueError, OSError) as exc:
            raise DataSourceError(f"Failed to read Excel {self.path}: {exc}") from exc


def source_for(path: Path | str, **kwargs) -> DataSource:
    """Pick a :class:`DataSource` from the file extension.

    Keeps callers from having to care whether the input is CSV or Excel, which
    matters here because the same analysis runs against both the raw workbook
    and the exported CSV variants.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        separator = kwargs.pop("separator", "\t" if suffix == ".tsv" else ",")
        return CsvSource(path, separator=separator, **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return ExcelSource(path, **kwargs)
    raise DataSourceError(
        f"Unsupported file type '{suffix}' for {path}. "
        f"Expected one of: .csv, .tsv, .txt, .xlsx, .xls, .xlsm"
    )


def write_frame(frame: pd.DataFrame, path: Path | str) -> Path:
    """Write ``frame`` to ``path``, choosing the writer from the extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
    elif suffix in {".xlsx", ".xlsm"}:
        frame.to_excel(path, index=False)
    elif suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        raise DataSourceError(f"Unsupported output type '{suffix}' for {path}")
    return path
