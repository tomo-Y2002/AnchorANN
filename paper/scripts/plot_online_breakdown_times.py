#!/usr/bin/env python3
"""Stacked bar chart on a linear time axis: AnchorANN, Amora, and Pacmann on SIFT1M WAN.

Input : paper/data/sift1m_wan_online_breakdown.csv
Output: paper/figures/OnlineBreakdownTimes.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper/figures/OnlineBreakdownTimes.pdf"

CSV = REPO / "paper/data/sift1m_wan_online_breakdown.csv"

COMPONENTS = ["client_s", "network_s", "server_s"]
LABELS = {
    "client_s": "Client",
    "network_s": "Network",
    "server_s": "Server",
}
COLORS = {"client_s": "#1baf7a", "network_s": "#eb6834", "server_s": "#2a78d6"}
COMMON_RECALL = 0.99
LABEL_MIN_PERCENT = 6


def format_total(seconds: float) -> str:
    return f"{seconds:.2f} s" if seconds >= 1 else f"{seconds * 1e3:.1f} ms"


def format_rounds(rounds: float) -> str:
    if rounds == int(rounds):
        return f"{int(rounds):,} rounds" if rounds != 1 else "1 round"
    return f"{rounds:.2f} rounds"


def format_method(method: str, recall: float) -> str:
    return method if recall == COMMON_RECALL else f"{method}*"


LABEL_MIN_MS = 300
TEXT_COLOR = "#0b0b0b"


def draw_times(ax: plt.Axes, df: pd.DataFrame) -> None:
    left = [0.0] * len(df)
    for c in COMPONENTS:
        for i, value in enumerate(df[c] * 1e3):
            ax.barh(i, value, left=left[i], height=0.62, color=COLORS[c], edgecolor="white", linewidth=0.4)
            if value >= LABEL_MIN_MS:
                ax.text(left[i] + value / 2, i, f"{value:.0f} ms", ha="center", va="center", fontsize=6.5,
                        color=TEXT_COLOR)
            left[i] += value
    for i, method in enumerate(df.index):
        row = df.loc[method]
        text = format_total(row["total_s"])
        if row["client_s"] * 1e3 < LABEL_MIN_MS:
            parts = " + ".join(f"{row[c] * 1e3:.1f}" for c in COMPONENTS)
            # Only the shortest bar has room for the component names, and it is drawn first.
            text += f"\n(client + network + server: {parts} ms)" if i == 0 else f"\n({parts} ms)"
        ax.text(left[i] + 15, i, text, va="center", fontsize=6.5, color=TEXT_COLOR)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(
        [f"{format_method(m, df.loc[m, 'one_recall'])}\n({format_rounds(df.loc[m, 'rounds'])})" for m in df.index],
        fontsize=6.5,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1950)
    ax.set_xticks(range(0, 1601, 400))
    ax.tick_params(axis="x", labelsize=7, length=2)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Online time per query [ms]", fontsize=7, labelpad=1)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)



ORDER = ["AnchorANN", "Amora", "Pacmann"]


def main() -> None:
    df = pd.read_csv(CSV).set_index("method").loc[ORDER]
    fig, ax = plt.subplots(figsize=(3.5, 1.55))
    draw_times(ax, df)
    # The modeled WAN time of Pacmann leaves its component labels to the right of a 1.5 s bar.
    ax.set_xlim(0, 2700)
    ax.set_xticks(range(0, 2001, 500))
    handles = [Patch(facecolor=COLORS[c], label=LABELS[c]) for c in COMPONENTS]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.5, frameon=False,
              handlelength=1.2, columnspacing=1.0, borderaxespad=0.2)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.21)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
