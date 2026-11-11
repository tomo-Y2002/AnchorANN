#!/usr/bin/env python3
"""Bar chart: communication rounds per query on SIFT1M WAN.

Only the round count is plotted. The PIR request count is not comparable across
methods, because one request fetches a vector in Pacmann, a cluster in
AnchorANN, and a partition in Amora.

Input : paper/data/sift1m_wan_rounds_pir.csv
Output: paper/figures/RoundsPIR.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "paper/data/sift1m_wan_rounds_pir.csv"
OUT = REPO / "paper/figures/RoundsPIR.pdf"

ORDER = ["AnchorANN", "Amora", "Pacmann"]
COLORS = {"AnchorANN": "#2a78d6", "Pacmann": "#eb6834", "Amora": "#1baf7a"}
LABELS = {"AnchorANN": "AnchorANN (proposed)", "Amora": "Amora", "Pacmann": "Pacmann"}


def main() -> None:
    df = pd.read_csv(CSV).set_index("method").loc[ORDER]
    fig, ax = plt.subplots(figsize=(3.5, 0.95))
    column = "rounds"
    for i, method in enumerate(df.index):
        value = df.loc[method, column]
        ax.barh(i, value, height=0.62, color=COLORS[method])
        ax.text(value + df[column].max() * 0.015, i, f"{value:,}", va="center", fontsize=7, color="#0b0b0b")
    ax.set_xlim(0, df[column].max() * 1.12)
    ax.set_xlabel("Communication rounds per query", fontsize=7, labelpad=1)
    ax.tick_params(axis="x", labelsize=6.5, length=2)
    ax.tick_params(axis="y", length=0, labelsize=7)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([LABELS[m] for m in df.index])
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.36, right=0.98, top=0.96, bottom=0.33)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)


if __name__ == "__main__":
    main()
