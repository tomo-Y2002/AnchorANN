#!/usr/bin/env python3
"""Recall--QPS of four first-1000-query SIFT1M ablations over modeled WAN.

Hollow markers are settings in which some queries lost clusters to PIR misses.

Input : paper/data/sift1m_ablation.csv
Output: paper/figures/Ablation.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "paper/data/sift1m_ablation.csv"
OUT = REPO / "paper/figures/Ablation.pdf"

STYLES = {
    "full": dict(label="Full", color="#eb6834", marker="o", zorder=4),
    "no-graph": dict(label="No graph", color="#2a78d6", marker="s", zorder=3),
    "no-batch": dict(label="No batch", color="#1baf7a", marker="P", zorder=3),
    "no-cluster-entry": dict(label="No cluster entry", color="#7a5fc9", marker="D", zorder=3),
}
LABELED = {"full": (4, 4), "no-batch": (4, 4), "no-cluster-entry": (4, 4)}


def plot(df):
    if set(df.setting) != {"WAN"} or set(df.queries) != {1000}:
        raise ValueError("requires WAN/1000-query measurements")
    if set(df.variant) != set(STYLES) or any(len(g) != 6 or set(g.m) != {4,8,16,32,64,128}
                                          for _, g in df.groupby("variant")):
        raise ValueError("requires all four variants and six m values")
    if (df.qps <= 0).any() or not df.qps.map(lambda v: 0 < v < float('inf')).all():
        raise ValueError("QPS must be finite and positive")
    fig, ax = plt.subplots(figsize=(3.5, 2.05))
    for variant, style in STYLES.items():
        g = df[df.variant == variant].sort_values("m")
        ax.plot(g.qps, g.one_recall, linewidth=1.2, markersize=3.6, **style)
        lost = g[g.missed_clusters > 0]
        ax.plot(lost.qps, lost.one_recall, linestyle="none", marker=style["marker"], markersize=3.6,
                markerfacecolor="white", markeredgecolor=style["color"], markeredgewidth=0.9, zorder=5)
        if variant in LABELED:
            for row in g.itertuples():
                ax.annotate(f"{row.m}", (row.qps, row.one_recall), xytext=LABELED[variant],
                            textcoords="offset points", fontsize=6, color="#55544f", va="center")
    ax.set_xscale("log")
    ax.set_xlim(df.qps.min()*0.8, df.qps.max()*1.4)
    ax.set_ylim(max(0, df.one_recall.min()-0.035), 1.025)
    ticks = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    ticks = [t for t in ticks if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.grid(True, which="major", color="#e6e5e0", linewidth=0.5)
    ax.grid(True, which="minor", axis="x", color="#f0efeb", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    ax.set_xlabel("QPS", fontsize=7, labelpad=1)
    ax.set_ylabel("1-Recall@10", fontsize=7, labelpad=2)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.legend(loc="lower left", ncol=2, fontsize=6.5, frameon=False, handlelength=2.0, labelspacing=0.25,
              columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.985, bottom=0.15)
    return fig, ax


def main() -> None:
    fig, _ = plot(pd.read_csv(CSV))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    plt.close(fig)


if __name__ == "__main__":
    main()
