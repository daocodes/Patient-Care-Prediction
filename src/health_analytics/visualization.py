"""Figure creation and output.

Two things this centralises:

* **Style.** Applied once, so all figures across all stages match.
* **Lifecycle.** Matplotlib figures are a process-global resource; forgetting
  ``plt.close()`` in a loop over 70 columns exhausts memory and triggers the
  "more than 20 figures have been opened" warning. :class:`FigureWriter` closes
  figures on the way out, including when the plotting code raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Select the non-interactive backend before pyplot is imported. The pipeline
# runs headless (CI, a server, a terminal over SSH) where no display exists.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import seaborn as sns  # noqa: E402

from .config import PlotConfig  # noqa: E402


def apply_style(config: PlotConfig) -> None:
    """Install the project's shared seaborn/matplotlib theme."""
    sns.set_theme(style=config.style, context=config.context)
    plt.rcParams.update(
        {
            "figure.figsize": config.figure_size,
            "figure.dpi": 100,
            "savefig.dpi": config.dpi,
            "savefig.bbox": "tight",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
        }
    )


@dataclass
class FigureWriter:
    """Creates figures, saves them into ``output_dir``, and cleans them up.

    Usage::

        with writer.figure("missingness.png", size=(16, 10)) as ax:
            sns.heatmap(frame.isna(), ax=ax)

    The file is written and the figure closed when the block exits, so callers
    never repeat the save/close boilerplate or leak a figure on an exception.
    """

    output_dir: Path
    config: PlotConfig

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def figure(
        self, filename: str, size: tuple[float, float] | None = None, title: str = ""
    ) -> Iterator[plt.Axes]:
        """Yield an Axes; save it to ``filename`` and close it afterwards."""
        fig, ax = plt.subplots(figsize=size or self.config.figure_size)
        try:
            if title:
                ax.set_title(title)
            yield ax
            self._save(fig, filename)
        finally:
            # `finally` rather than `else`: a failed plot must still release the
            # figure, otherwise one bad column leaks memory for the whole run.
            plt.close(fig)

    @contextmanager
    def grid(
        self, filename: str, rows: int, cols: int, size: tuple[float, float]
    ) -> Iterator[list[plt.Axes]]:
        """Yield a flat list of Axes for a subplot grid, then save and close."""
        fig, axes = plt.subplots(rows, cols, figsize=size)
        try:
            yield list(axes.flatten())
            fig.tight_layout()
            self._save(fig, filename)
        finally:
            plt.close(fig)

    def save_existing(self, fig: plt.Figure, filename: str) -> Path:
        """Save a figure built by a library that creates its own.

        Seaborn's ``clustermap`` and ``pairplot`` return their own grid object
        instead of drawing onto a supplied Axes, so they cannot use
        :meth:`figure`. The original code called the module-level
        ``plt.savefig`` after these, which saves whichever figure happens to be
        current -- frequently the wrong one, and the reason the old cluster map
        came out untitled.
        """
        try:
            path = self._save(fig, filename)
        finally:
            plt.close(fig)
        return path

    def _save(self, fig: plt.Figure, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.config.dpi, bbox_inches="tight")
        return path
