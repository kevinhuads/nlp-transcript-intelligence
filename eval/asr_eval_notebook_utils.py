from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch


### Dataset ###


def list_eval_csvs(results_root: str, dataset: str) -> List[str]:
    pattern = os.path.join(results_root, dataset, "asr_eval_*.csv")
    paths = sorted(glob.glob(pattern, recursive=False))
    return [p for p in paths if os.path.getsize(p) > 0]


def load_eval_df(results_root: str, dataset: str) -> pd.DataFrame:
    csv_paths = list_eval_csvs(results_root, dataset)
    frames = [pd.read_csv(p) for p in csv_paths]
    df = pd.concat(frames, ignore_index=True)

    num_cols = ["wer", "substitutions", "deletions", "insertions", "ref_words", "hyp_words", "asr_seconds"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if "audio_path" in df.columns:
        df = df.drop(columns=["audio_path"])

    df["model"] = df["model"].astype(str)
    df["id"] = df["id"].astype(str)

    df["ref_words"] = df["ref_words"].fillna(0).astype(int)
    for c in ["substitutions", "deletions", "insertions", "hyp_words"]:
        df[c] = df[c].fillna(0).astype(int)

    df["wer"] = df["wer"].astype(float)
    df["asr_seconds"] = df["asr_seconds"].astype(float)
    df["err_words"] = df["substitutions"] + df["deletions"] + df["insertions"]

    return df


def summarize_eval_df(df: pd.DataFrame) -> pd.DataFrame:
    model_summary = (
        df.groupby("model", as_index=False)
          .agg(
              n_utts=("id", "nunique"),
              total_ref_words=("ref_words", "sum"),
              total_S=("substitutions", "sum"),
              total_D=("deletions", "sum"),
              total_I=("insertions", "sum"),
              total_err=("err_words", "sum"),
              mean_item_wer=("wer", "mean"),
              median_item_wer=("wer", "median"),
              p90_item_wer=("wer", lambda x: float(np.percentile(x, 90))),
              total_asr_seconds=("asr_seconds", "sum"),
              median_asr_seconds=("asr_seconds", "median"),
              p90_asr_seconds=("asr_seconds", lambda x: float(np.percentile(x, 90))),
          )
    )

    model_summary["corpus_wer"] = model_summary["total_err"] / model_summary["total_ref_words"]
    model_summary["sec_per_ref_word"] = model_summary["total_asr_seconds"] / model_summary["total_ref_words"]

    model_summary["S_per_word"] = model_summary["total_S"] / model_summary["total_ref_words"]
    model_summary["D_per_word"] = model_summary["total_D"] / model_summary["total_ref_words"]
    model_summary["I_per_word"] = model_summary["total_I"] / model_summary["total_ref_words"]

    return model_summary.sort_values("corpus_wer").reset_index(drop=True)


def make_order(model_summary: pd.DataFrame) -> List[str]:
    return model_summary["model"].astype(str).tolist()


def make_default_palette(shade_factor: float = 0.55) -> ModelPalette:
    return ModelPalette(
        family_fn=model_family,
        family_base=family_colors,
        family_label=family_labels,
        shade_factor=shade_factor,
    )
    
    
def oracle_win_counts(df: pd.DataFrame, order: Iterable[str], tie_mode: str = "fractional") -> pd.DataFrame:
    models = [str(m) for m in order]
    pivot = df.pivot_table(index="id", columns="model", values="wer", aggfunc="mean").reindex(columns=models)

    row_min = pivot.min(axis=1)
    win_mask = pivot.eq(row_min, axis=0)

    if tie_mode == "first":
        winners = pivot.idxmin(axis=1)
        counts = winners.value_counts().reindex(models).fillna(0).astype(int)
        out = pd.DataFrame({"model": models, "n_wins": counts.to_numpy(dtype=int)})
        out["win_share"] = out["n_wins"] / float(len(pivot))
        return out.sort_values(["win_share", "n_wins", "model"], ascending=[False, False, True]).reset_index(drop=True)

    if tie_mode == "strict":
        is_tie = (win_mask.sum(axis=1) > 1)
        strict = win_mask.loc[~is_tie]
        counts = strict.sum(axis=0).reindex(models).fillna(0).astype(int)
        denom = float(len(strict)) if len(strict) else 1.0
        out = pd.DataFrame({"model": models, "n_wins": counts.to_numpy(dtype=int)})
        out["win_share"] = out["n_wins"] / denom
        return out.sort_values(["win_share", "n_wins", "model"], ascending=[False, False, True]).reset_index(drop=True)

    if tie_mode == "fractional":
        frac = win_mask.div(win_mask.sum(axis=1), axis=0).sum(axis=0).reindex(models).fillna(0.0)
        out = pd.DataFrame({"model": models, "n_wins": frac.to_numpy(dtype=float)})
        out["win_share"] = out["n_wins"] / float(len(pivot))
        return out.sort_values(["win_share", "n_wins", "model"], ascending=[False, False, True]).reset_index(drop=True)

    raise ValueError("tie_mode must be one of: 'fractional', 'strict', 'first'")




def oracle_corpus_wer(df: pd.DataFrame, order: Iterable[str], tie_mode: str = "first") -> float:
    models = [str(m) for m in order]
    pivot = df.pivot_table(index="id", columns="model", values="wer", aggfunc="mean").reindex(columns=models)
    row_min = pivot.min(axis=1)
    win_mask = pivot.eq(row_min, axis=0)

    if tie_mode == "first":
        winners = pivot.idxmin(axis=1)
        chosen = df.merge(pd.DataFrame({"id": pivot.index, "oracle_model": winners}), on="id", how="inner")
        chosen = chosen[chosen["model"].astype(str) == chosen["oracle_model"].astype(str)]
        return float(chosen["err_words"].sum() / chosen["ref_words"].sum())

    if tie_mode == "strict":
        is_tie = (win_mask.sum(axis=1) > 1)
        winners = pivot.loc[~is_tie].idxmin(axis=1)
        chosen = df.merge(pd.DataFrame({"id": winners.index, "oracle_model": winners}), on="id", how="inner")
        chosen = chosen[chosen["model"].astype(str) == chosen["oracle_model"].astype(str)]
        return float(chosen["err_words"].sum() / chosen["ref_words"].sum())

    raise ValueError("tie_mode must be 'first' or 'strict'")

### Plotting ### 

sns.set_theme(
    style="darkgrid",
    rc={
        "figure.facecolor": "#0d1b2a",
        "axes.facecolor": "#0d1b2a",
        "axes.edgecolor": "#cccccc",
        "grid.color": "#2a3f5f",
        "axes.labelcolor": "#ffffff",
        "text.color": "#ffffff",
        "xtick.color": "#ffffff",
        "ytick.color": "#ffffff",
    },
    palette="deep",
)
sns.set_palette(sns.color_palette(["#1f4aa8", "#2d6cdf", "#4ea3ff", "#7bc7ff"]))


hue_colors = [
    "#F8766D",
    "#7CAE00",
    "#00BFC4",
    "#C77CFF",
    "#00BA38",
    "#619CFF",
    "#FF61C3",
    "#00A9FF",
    "#E76BF3",
    "#B79F00",
]

family_labels: Dict[str, str] = {
    "ssl_ctc": "Self-supervised CTC",
    "nemo_ctc": "NeMo CTC",
    "whisper": "Whisper",
    "transducer": "Transducer",
    "other": "Other",
}

family_colors: Dict[str, str] = {k: c for k, c in zip(family_labels.keys(), hue_colors)}


def model_family(model: str) -> str:
    m = str(model).lower()
    if m.startswith(("tiny", "base", "small", "medium", "large")) or "distil" in m or "turbo" in m:
        return "whisper"
    if m.startswith(("wav2vec2", "hubert", "mms")):
        return "ssl_ctc"
    if "transducer" in m or "rnnt" in m:
        return "transducer"
    if m.startswith(("conformer", "citrinet", "quartznet", "jasper")):
        return "nemo_ctc"
    return "other"


def _hex_to_rgb01(h: str) -> Tuple[float, float, float]:
    h = h.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _rgb01_to_hex(rgb: Tuple[float, float, float]) -> str:
    r = int(round(max(0.0, min(1.0, rgb[0])) * 255))
    g = int(round(max(0.0, min(1.0, rgb[1])) * 255))
    b = int(round(max(0.0, min(1.0, rgb[2])) * 255))
    return f"#{r:02x}{g:02x}{b:02x}"


def _mix_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = _hex_to_rgb01(a)
    br, bg, bb = _hex_to_rgb01(b)
    t = max(0.0, min(1.0, float(t)))
    return _rgb01_to_hex((ar * (1 - t) + br * t, ag * (1 - t) + bg * t, ab * (1 - t) + bb * t))


def _stable_unit_interval(s: str) -> float:
    d = md5(s.encode("utf-8")).digest()
    n = int.from_bytes(d[:8], "big")
    return n / float(2**64 - 1)


@dataclass(frozen=True)
class ModelPalette:
    family_fn: Callable[[str], str]
    family_base: Dict[str, str]
    family_label: Dict[str, str]
    shade_factor: float = 0.55
    other_color: str = "#9CA3AF"

    def family_of(self, model: str) -> str:
        return self.family_fn(str(model))

    def base_of(self, family: str) -> str:
        return self.family_base.get(family, self.other_color)

    def label_of(self, family: str) -> str:
        return self.family_label.get(family, family)

    def color_of(self, model: str) -> str:
        m = str(model)
        fam = self.family_of(m)
        base = self.base_of(fam)
        dark = _mix_hex(base, "#000000", self.shade_factor)
        t = _stable_unit_interval(f"{fam}::{m.lower()}")
        t = 0.15 + 0.85 * t
        return _mix_hex(base, dark, t)

    def filter_models(self, order: Iterable[str], families: Optional[Iterable[str]] = None) -> List[str]:
        models = [str(m) for m in order]
        if families is None:
            return models
        famset = set(families)
        return [m for m in models if self.family_of(m) in famset]

    def legend_handles(self, models: Iterable[str]) -> List[Patch]:
        ms = [str(m) for m in models]
        present = {self.family_of(m) for m in ms}
        handles: List[Patch] = []
        for fam in self.family_label.keys():
            if fam in present:
                handles.append(
                    Patch(
                        facecolor=self.base_of(fam),
                        edgecolor="white",
                        label=self.label_of(fam),
                    )
                )
        return handles


def make_palette(
    family_fn: Callable[[str], str] = model_family,
    family_base: Optional[Dict[str, str]] = None,
    family_label: Optional[Dict[str, str]] = None,
    shade_factor: float = 0.55,
    other_color: str = "#9CA3AF",
) -> ModelPalette:
    if family_base is None:
        family_base = family_colors
    if family_label is None:
        family_label = family_labels
    return ModelPalette(
        family_fn=family_fn,
        family_base=family_base,
        family_label=family_label,
        shade_factor=shade_factor,
        other_color=other_color,
    )


def plot_corpus_wer_barh(
    model_summary: pd.DataFrame,
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    ax=None,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    use_order = palette.filter_models(order, families)
    plot_df = model_summary[model_summary["model"].astype(str).isin(use_order)].copy()
    plot_df = plot_df.sort_values("corpus_wer").reset_index(drop=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(plot_df))))
    else:
        fig = ax.figure

    models = plot_df["model"].astype(str).tolist()
    ax.barh(
        models,
        plot_df["corpus_wer"].to_numpy(dtype=float),
        color=[palette.color_of(m) for m in models],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.set_xlabel("corpus wer")
    ax.grid(True, axis="x", alpha=0.5)
    ax.legend(
        handles=palette.legend_handles(use_order),
        loc="lower right",
        frameon=True,
        facecolor=ax.get_facecolor(),
        edgecolor="white",
        fontsize=11,
    )
    fig.tight_layout()

    if return_data:
        return fig, ax, {"plot_df": plot_df, "use_order": use_order}
    return fig, ax


def plot_latency_vs_wer(
    model_summary: pd.DataFrame,
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    ax=None,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    use_order = palette.filter_models(order, families)
    plot_df = model_summary[model_summary["model"].astype(str).isin(use_order)].copy()
    plot_df["model"] = plot_df["model"].astype(str)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    for _, r in plot_df.iterrows():
        m = str(r["model"])
        ax.scatter(
            float(r["median_asr_seconds"]),
            float(r["corpus_wer"]),
            color=palette.color_of(m),
            edgecolor="white",
            linewidth=0.8,
        )
        ax.annotate(m, (float(r["median_asr_seconds"]), float(r["corpus_wer"])), fontsize=9, alpha=0.9)

    ax.set_xlabel("median asr seconds per utterance")
    ax.set_ylabel("corpus wer")
    ax.grid(True, alpha=0.5)
    ax.legend(
        handles=palette.legend_handles(use_order),
        loc="upper right",
        frameon=True,
        facecolor=ax.get_facecolor(),
        edgecolor="white",
        fontsize=11,
    )
    fig.tight_layout()

    if return_data:
        return fig, ax, {"plot_df": plot_df, "use_order": use_order}
    return fig, ax


def plot_wer_boxplot(
    df: pd.DataFrame,
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    ax=None,
    showfliers: bool = False,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    use_order = palette.filter_models(order, families)
    work = df[df["model"].astype(str).isin(use_order)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(use_order))))
    else:
        fig = ax.figure

    data = [work.loc[work["model"].astype(str) == m, "wer"].to_numpy(dtype=float) for m in use_order]
    bp = ax.boxplot(data, vert=False, tick_labels=use_order, showfliers=showfliers, patch_artist=True)

    for i, m in enumerate(use_order):
        c = palette.color_of(m)
        bp["boxes"][i].set_facecolor(c)
        bp["boxes"][i].set_edgecolor("white")
        bp["boxes"][i].set_linewidth(1.6)
        bp["medians"][i].set_color("white")
        bp["medians"][i].set_linewidth(1.8)
        for w in (bp["whiskers"][2 * i], bp["whiskers"][2 * i + 1]):
            w.set_color("white")
            w.set_linewidth(1.2)
        for cap in (bp["caps"][2 * i], bp["caps"][2 * i + 1]):
            cap.set_color("white")
            cap.set_linewidth(1.2)

    ax.set_xlabel("utterance wer")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(
        handles=palette.legend_handles(use_order),
        loc="lower right",
        frameon=True,
        facecolor=ax.get_facecolor(),
        edgecolor="white",
        fontsize=9,
    )
    fig.tight_layout()

    if return_data:
        return fig, ax, {"work_df": work, "use_order": use_order}
    return fig, ax


def plot_wer_cdf(
    df: pd.DataFrame,
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    ax=None,
    max_percentile: float = 99.0,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    use_order = palette.filter_models(order, families)
    work = df[df["model"].astype(str).isin(use_order)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if len(work):
        xmax = float(np.percentile(work["wer"].to_numpy(dtype=float), max_percentile))
        xmax = max(0.5, xmax)
    else:
        xmax = 0.5

    xs = np.linspace(0, xmax, 400)

    cdf = pd.DataFrame(index=xs)
    for m in use_order:
        w = work.loc[work["model"].astype(str) == m, "wer"].to_numpy(dtype=float)
        if w.size == 0:
            continue
        w = np.sort(w)
        ys = np.searchsorted(w, xs, side="right") / float(len(w))
        cdf[m] = ys
        ax.plot(xs, ys, label=m, color=palette.color_of(m), linewidth=1.8)

    ax.set_xlabel("utterance wer")
    ax.set_ylabel("cdf")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if return_data:
        return fig, ax, {"xs": xs, "cdf": cdf, "work_df": work, "use_order": use_order}
    return fig, ax


def wer_by_length_pivot(
    df: pd.DataFrame,
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    bins: Optional[List[float]] = None,
    labels: Optional[List[str]] = None,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    if bins is None:
        bins = [-0.1, 2, 5, 10, 20, 50, 10_000]
    if labels is None:
        labels = ["<=2", "3-5", "6-10", "11-20", "21-50", ">50"]

    use_order = palette.filter_models(order, families)
    work = df[df["model"].astype(str).isin(use_order)].copy()
    work["len_bin"] = pd.cut(work["ref_words"], bins=bins, labels=labels)

    len_summary = (
        work.groupby(["model", "len_bin"], observed=False)
        .agg(total_ref_words=("ref_words", "sum"), total_err=("err_words", "sum"))
        .reset_index()
    )
    len_summary["corpus_wer_in_bin"] = len_summary["total_err"] / len_summary["total_ref_words"]

    pivot = (
        len_summary.pivot(index="len_bin", columns="model", values="corpus_wer_in_bin")
        .reindex(labels)
        .loc[:, use_order]
    )

    if return_data:
        return pivot, labels, {"len_summary": len_summary, "work_df": work, "use_order": use_order}
    return pivot, labels


def plot_wer_by_length(
    pivot: pd.DataFrame,
    labels: List[str],
    order: Iterable[str],
    palette: ModelPalette,
    families: Optional[Iterable[str]] = None,
    ax=None,
    return_data: bool = False,
) -> Tuple[Any, ...]:
    use_order = palette.filter_models(order, families)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    x = np.arange(len(labels))
    for m in use_order:
        if m not in pivot.columns:
            continue
        ax.plot(
            x,
            pivot[m].to_numpy(dtype=float),
            marker="o",
            label=m,
            color=palette.color_of(m),
            linewidth=1.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("ref words bin")
    ax.set_ylabel("corpus wer within bin")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if return_data:
        return fig, ax, {"use_order": use_order}
    return fig, ax
