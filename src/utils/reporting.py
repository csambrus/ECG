from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR, PLOT_DPI, PLOT_INLINE, PLOT_SAVE, PLOT_SHOW


def _is_running_in_notebook() -> bool:
    """
    Megpróbáljuk eldönteni, hogy Jupyter / IPython notebook környezetben futunk-e.
    """
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False

        shell_name = ip.__class__.__name__
        return shell_name in {"ZMQInteractiveShell"}
    except Exception:
        return False


def _display_inline(fig) -> bool:
    """
    Notebookban inline megjelenítés.
    True-val tér vissza, ha sikerült.
    """
    try:
        from IPython.display import display

        display(fig)
        return True
    except Exception:
        return False


def plot_save_or_show(
    fig,
    out_path: str | Path | None = None,
    show: bool | None = None,
    save: bool | None = None,
    inline: bool | None = None,
    close: bool = True,
) -> Path | None:
    """
    Egységes plot kezelés.

    Funkciók:
    - mentés fájlba
    - notebook inline megjelenítés
    - klasszikus plt.show() script / CLI módban
    - kontrollált lezárás

    Prioritás:
    1) mentés, ha kell
    2) ha notebook + inline engedélyezett -> display(fig)
    3) különben ha show engedélyezett -> plt.show()

    Visszatér:
    - a mentett fájl Path-jával, ha történt mentés
    - különben None
    """
    effective_show = PLOT_SHOW if show is None else show
    effective_save = PLOT_SAVE if save is None else save
    effective_inline = PLOT_INLINE if inline is None else inline

    saved_path: Path | None = None

    if effective_save and out_path is not None:
        saved_path = Path(out_path)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(saved_path, dpi=PLOT_DPI, bbox_inches="tight")
        print(f"[FIGURE] {saved_path}")

    displayed_inline = False
    if effective_inline:
        displayed_inline = _display_inline(fig)

    if effective_show and not displayed_inline:
        plt.show()

    if close:
        plt.close(fig)

    return saved_path


class Reporter:
    """
    Egységes riportoló segéd.

    Használat:
        reporter = Reporter(out_dir)
        with reporter.figure("my_plot") as (fig, ax):
            ax.plot(...)
    """

    def __init__(
        self,
        out_dir: Path | str | None = None,
        show: bool | None = None,
        save: bool | None = None,
        inline: bool | None = None,
        close: bool = True,
        verbose: bool = True,
    ):
        self.out_dir = Path(out_dir or OUTPUT_DIR)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.show = PLOT_SHOW if show is None else show
        self.save = PLOT_SAVE if save is None else save
        self.inline = PLOT_INLINE if inline is None else inline
        self.close = close
        self.verbose = verbose

    def figure(
        self,
        name: str,
        figsize: tuple[float, float] = (12, 8),
        tight_layout: bool = True,
    ):
        return FigureContext(
            reporter=self,
            name=name,
            figsize=figsize,
            tight_layout=tight_layout,
        )

    def save_df(
        self,
        df,
        name: str,
        index: bool = False,
        print_df: bool = False,
    ) -> Path:
        path = self.out_dir / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=index)

        if self.verbose:
            print(f"[TABLE] {path}")

        if print_df:
            try:
                from IPython.display import display

                display(df)
            except Exception:
                print(df.to_string(index=index))

        return path

    def save_text(self, text: str, name: str, suffix: str = ".txt") -> Path:
        path = self.out_dir / f"{name}{suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

        if self.verbose:
            print(f"[TEXT] {path}")

        return path

    def save_json(self, data: Any, name: str) -> Path:
        import json

        path = self.out_dir / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.verbose:
            print(f"[JSON] {path}")

        return path


class FigureContext:
    def __init__(
        self,
        reporter: Reporter,
        name: str,
        figsize: tuple[float, float],
        tight_layout: bool = True,
    ):
        self.reporter = reporter
        self.name = name
        self.figsize = figsize
        self.tight_layout = tight_layout
        self.fig = None
        self.ax = None
        self.saved_path: Path | None = None

    def __enter__(self):
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        return self.fig, self.ax

    def __exit__(self, exc_type, exc, tb):
        if self.fig is None:
            return False

        if exc_type is not None:
            plt.close(self.fig)
            return False

        if self.tight_layout:
            try:
                self.fig.tight_layout()
            except Exception:
                pass

        out_path = self.reporter.out_dir / f"{self.name}.png"

        self.saved_path = plot_save_or_show(
            fig=self.fig,
            out_path=out_path,
            show=self.reporter.show,
            save=self.reporter.save,
            inline=self.reporter.inline,
            close=self.reporter.close,
        )

        return False