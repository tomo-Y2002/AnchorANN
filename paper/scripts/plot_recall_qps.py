#!/usr/bin/env python3
"""Recall--QPS panels of the ICDE evaluation in one figure.

Rows are (a) LAN online, (b) WAN online, (c) LAN amortized, (d) WAN amortized;
columns are the datasets. Rows of the same QPS kind share the axis ranges.
A dataset with no finished experiment stays as an empty panel.

Input : paper/data/recall_qps.csv, paper/data/recall_qps_amortized.csv
Output: paper/figures/RecallQPS.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "paper/data/recall_qps.csv"
AMORTIZED_CSV = REPO / "paper/data/recall_qps_amortized.csv"
FIGURES = REPO / "paper/figures"

DATASETS = ["SIFT1M", "DEEP1M", "GIST1M", "MSMARCO"]
STYLES = {
    "AnchorANN": dict(color="#2a78d6", marker="s", zorder=5),
    "Pacmann": dict(color="#eb6834", marker="o", zorder=3),
    "Amora": dict(color="#1baf7a", marker="P", zorder=3),
    "NSG": dict(color="#a8842c", marker="^", zorder=3),
}
# NSG searches the plaintext query on the server, so it is an upper bound rather than a private method.
LABELS = {"NSG": "NSG (oracle, non-private)"}


ROWS = [
    ("a", "LAN", "online"),
    ("b", "WAN", "online"),
    ("c", "LAN", "amortized"),
    ("d", "WAN", "amortized"),
]
XLIMS = {
    "online": ((0.03, 2000), (0.1, 1, 10, 100, 1000)),
    "amortized": ((0.01, 2000), (0.01, 0.1, 1, 10, 100, 1000)),
}


def plot(online: pd.DataFrame, amortized: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ROWS), len(DATASETS), figsize=(7.16, 4.6), sharey=True)
    for (row, setting, kind), row_axes in zip(ROWS, axes):
        df = online if kind == "online" else amortized
        xlim, xticks = XLIMS[kind]
        for i, (ax, dataset) in enumerate(zip(row_axes, DATASETS), start=1):
            panel = df[(df.setting == setting) & (df.dataset == dataset)]
            for method, style in STYLES.items():
                s = panel[panel.method == method].sort_values("order")
                ax.plot(s.qps, s.one_recall, label=LABELS.get(method, method), linewidth=1.0, markersize=2.8, **style)
            if panel.empty:
                ax.text(0.5, 0.5, "Not available yet", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6.5, color="#6b6a65")
            ax.set_xscale("log")
            ax.set_xlim(*xlim)
            ax.set_ylim(0.2, 1.03)
            ax.set_xticks(xticks)
            # Minor grid lines at 2-9 times each power of ten show that the axis is logarithmic.
            ax.grid(True, which="major", color="#e6e5e0", linewidth=0.5)
            ax.grid(True, which="minor", axis="x", color="#f0efeb", linewidth=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.tick_params(which="minor", length=1)
            # The panel title goes below the axis, after the axis label.
            ax.set_xlabel(f"QPS\n({row}-{i}) {dataset}, {setting}, {kind}", fontsize=6.5, labelpad=1,
                          linespacing=1.5)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
        row_axes[0].set_ylabel("1-Recall@10", fontsize=6.5, labelpad=1)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(STYLES), fontsize=7, frameon=False,
               handlelength=2.0, columnspacing=1.4, borderaxespad=0.1)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.955, bottom=0.095, wspace=0.08, hspace=0.66)
    fig.savefig(FIGURES / "RecallQPS.pdf")


def main() -> None:
    plot(pd.read_csv(CSV), pd.read_csv(AMORTIZED_CSV))


if __name__ == "__main__":
    main()
