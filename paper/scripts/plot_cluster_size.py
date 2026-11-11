#!/usr/bin/env python3
"""WAN online Recall--QPS of each SIFT1M batch-PIR cluster size.

Thin gray lines join the points that share m, so sizes compare at the same setting.

Input : paper/data/sift1m_cluster_size.csv
Output: paper/figures/ClusterSize.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LogLocator, NullFormatter, ScalarFormatter

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "paper/data/sift1m_cluster_size.csv"
OUT = REPO / "paper/figures/ClusterSize.pdf"

STYLES = {
    16: dict(color="#2a78d6", marker="s"),
    32: dict(color="#eb6834", marker="o"),
    64: dict(color="#1baf7a", marker="P"),
    125: dict(color="#7a5fc9", marker="D"),
}


def load_data(path=CSV):
    df = pd.read_csv(path)
    expected = {(s, m) for s in STYLES for m in (4, 8, 16, 32, 64, 128)}
    if len(df) != 24 or set(zip(df.s, df.m)) != expected:
        raise ValueError("requires all 24 cluster-size points")
    if not ((df.setting == "WAN") & (df.queries == 1000) & (df.k == 10) &
            (df.ef == 128) & (df.rpcs_per_query == 1) & (df.qps > 0) &
            df.one_recall.between(0, 1) & (df.K == 1_000_000 // df.s)).all():
        raise ValueError("requires WAN batch-PIR, first1000 queries, k10 and L128")
    return df


def plot(df, output=OUT):
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    for m, group in df.groupby("m"):
        group = group.sort_values("qps")
        ax.plot(group.qps, group.one_recall, color="#b9b8b2", linewidth=0.7, linestyle=":", zorder=1)
        top = group.iloc[-1]
        ax.annotate(f"$m={m}$", (top.qps, top.one_recall), xytext=(0, 7) if m == 128 else (4, -1),
                    textcoords="offset points", fontsize=6.5, color="#55544f", va="center")
    for s, style in STYLES.items():
        g = df[df.s == s].sort_values("m")
        ax.plot(g.qps, g.one_recall, label=f"$s={s}$, $K={g.K.iloc[0]:,}$", linewidth=1.2, markersize=3.6, zorder=3, **style)
    ax.set_xscale("log")
    ax.set_xlim(df.qps.min() / 1.25, df.qps.max() * 1.3)
    ax.set_ylim(max(0, df.one_recall.min() - 0.07), 1.04)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # Minor grid lines at 2-9 times each power of ten show that the axis is logarithmic.
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.5)
    ax.grid(True, which="minor", axis="x", color="#f0efeb", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.set_xlabel("WAN online QPS", fontsize=7, labelpad=1)
    ax.set_ylabel("1-Recall@10", fontsize=7, labelpad=2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False, handlelength=2.0, labelspacing=0.25, borderaxespad=0.2)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.985, bottom=0.15)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.03)
    return fig, ax


def main() -> None:
    fig, _ = plot(load_data())
    plt.close(fig)


if __name__ == "__main__":
    main()
