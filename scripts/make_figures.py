"""Re-draw the report's key scores and matrices from the CSV files in data/.

Usage (from the repository root):
    python scripts/make_figures.py

Requires pandas and matplotlib. Output: PNG files in figures/.
"""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# Palette (light surface)
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e4e3df"
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#b9b8b2"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def wrap(s, n):
    return "\n".join(textwrap.wrap(s, n))


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


# 1. Degree of freedom (open vs closed system) -------------------------------
def fig_degree_of_freedom():
    df = pd.read_csv(DATA / "degree_of_freedom_items.csv")
    df = df.iloc[::-1]
    colors = df["classification_in_report"].map(
        {"Strength": BLUE, "Weakness": ORANGE, "Neutral": GRAY})
    fig, ax = plt.subplots(figsize=(8.5, 8))
    ax.barh([f"{i}. {c}" for i, c in zip(df["item"], df["characteristic"])],
            df["weighted_average_as_reported"], color=colors, height=0.65)
    for x, lab, ha, dx in ((140, "weakness if < 140", "right", -2), (170, "strength if > 170", "left", 2)):
        ax.axvline(x, color=INK2, lw=1, ls="--")
        ax.text(x + dx, 23.9, lab, fontsize=8, color=INK2, ha=ha, va="bottom")
    ax.set_xlabel("Weighted average score (as reported)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRAY)]
    ax.legend(handles, ["Strength", "Weakness", "Neither"], loc="lower right", frameon=False,
              title="Classification in the report", title_fontsize=8, fontsize=8)
    ax.set_title("Open vs. closed system: score per characteristic\n"
                 "Overall degree of freedom = 75.63 (5 respondents; CEO and product manager weighted x2)",
                 loc="left", fontsize=11)
    save(fig, "fig1_degree_of_freedom.png")


# 2. Organisational life cycle (Adizes) ---------------------------------------
def fig_life_cycle():
    df = pd.read_csv(DATA / "life_cycle_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, comp, title in zip(axes, ["Stage comparison", "Within adolescence"],
                               ["Which stage fits best?", "Adolescence and its pathologies"]):
        d = df[df["comparison"] == comp]
        top = d["share_percent"].max()
        cols = [BLUE if v == top else GRAY for v in d["share_percent"]]
        bars = ax.bar([wrap(s, 18) for s in d["stage"]], d["share_percent"], color=cols, width=0.55)
        for b, sh, m in zip(bars, d["share_percent"], d["mean_score"]):
            ax.text(b.get_x() + b.get_width() / 2, sh + 0.8, f"{sh:.2f}%\n(mean {m})",
                    ha="center", fontsize=9, color=INK)
        ax.set_ylim(0, 48)
        ax.set_ylabel("Share of total mean score (%)")
        ax.set_title(title, loc="left", fontsize=11)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    fig.suptitle("Organisational life cycle (Adizes): questionnaire results", x=0.01, ha="left",
                 fontsize=12, y=1.03)
    save(fig, "fig2_life_cycle.png")


# 3. Porter's five forces -------------------------------------------------------
def fig_porter():
    df = pd.read_csv(DATA / "porter_five_forces.csv")
    order = ["Barriers to entry", "Rivalry among existing competitors", "Bargaining power of buyers",
             "Threat of substitutes", "Bargaining power of suppliers"]
    g = df.groupby("force")["score"].agg(["mean", "count"]).reindex(order)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    y = [f"{f}\n({int(n)} questions)" for f, n in zip(g.index, g["count"])]
    ax.barh(y[::-1], g["mean"][::-1], color=BLUE, height=0.55)
    for yi, v in zip(range(len(g)), g["mean"][::-1]):
        ax.text(v + 0.05, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 5)
    ax.set_xlabel("Mean of item scores (0 to 5)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Porter's five forces: mean item score per force\n"
                 "(means computed from the report's item scores)", loc="left", fontsize=11)
    save(fig, "fig3_porter_five_forces.png")


# 4. IEFE: highest weighted factors per group -----------------------------------
def fig_iefe_top():
    df = pd.read_csv(DATA / "iefe_factors.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, t in zip(axes.flat, ["Strength", "Weakness", "Opportunity", "Threat"]):
        d = df[df["type"] == t]
        if t == "Weakness":  # most serious weaknesses = high importance, low status
            d = d.sort_values(["importance_0_10", "status_0_4"], ascending=[False, True]).head(6)
            val, lab = d["importance_0_10"], "Importance (0-10)"
            sub = "Most important weaknesses (status 0 = very serious)"
            txt = [f"imp {i}, status {s}" for i, s in zip(d["importance_0_10"], d["status_0_4"])]
        elif t == "Threat":
            d = d.sort_values(["importance_0_10", "status_0_4"], ascending=[False, True]).head(6)
            val, lab = d["importance_0_10"], "Importance (0-10)"
            sub = "Most important threats (status 0-4 = ability to respond)"
            txt = [f"imp {i}, status {s}" for i, s in zip(d["importance_0_10"], d["status_0_4"])]
        else:
            d = d.sort_values("weighted_score", ascending=False).head(6)
            val, lab = d["weighted_score"], "Weighted score (importance x status, max 40)"
            sub = f"Top {'opportunities' if t == 'Opportunity' else 'strengths'} by weighted score"
            txt = [str(v) for v in d["weighted_score"]]
        color = BLUE if t in ("Strength", "Opportunity") else ORANGE
        names = [f"{i}: " + wrap(f, 48) for i, f in zip(d["id"], d["factor"])]
        ax.barh(names[::-1], list(val)[::-1], color=color, height=0.6)
        for yi, (v, s) in enumerate(zip(list(val)[::-1], txt[::-1])):
            ax.text(v + 0.3, yi, s, va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, 44 if "Weighted" in lab else 13)
        ax.set_xlabel(lab, fontsize=9)
        ax.set_title(sub, loc="left", fontsize=10.5)
        ax.tick_params(axis="y", length=0, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
    fig.suptitle("Internal and external factor evaluation (IEFE): the factors that matter most",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    save(fig, "fig4_iefe_key_factors.png")


# 5. Internal-External matrix ------------------------------------------------------
def fig_ie_matrix():
    s = pd.read_csv(DATA / "iefe_summary.csv").set_index("factor_group")["percentage"]
    ip, ep = s["Internal (S+W)"], s["External (O+T)"]
    regions = {  # (col from left = internal high->low, row from top = external high->low)
        (0, 0): ("I", BLUE), (1, 0): ("II", AQUA), (0, 1): ("IV", AQUA), (2, 0): ("III", GRAY),
        (1, 1): ("V", GRAY), (0, 2): ("VII", GRAY), (2, 1): ("VI", ORANGE), (1, 2): ("VIII", ORANGE),
        (2, 2): ("IX", ORANGE)}
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    for (c, r), (lab, col) in regions.items():
        x0, y0 = 100 - (c + 1) * 100 / 3, 100 - (r + 1) * 100 / 3
        ax.add_patch(plt.Rectangle((x0, y0), 100 / 3, 100 / 3, facecolor=col, alpha=0.18,
                                   edgecolor=SURFACE, lw=3))
        ax.text(x0 + 3, y0 + 100 / 3 - 4, lab, fontsize=11, color=INK2, va="top")
    ax.scatter([ip], [ep], s=140, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"The Company\ninternal {ip:.2f}%\nexternal {ep:.2f}%", (ip, ep), xytext=(ip + 9, ep + 11),
                fontsize=9.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlim(100, 0)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 33, 66, 100])
    ax.set_yticks([0, 33, 66, 100])
    ax.set_xlabel("Internal key factors (%)")
    ax.set_ylabel("External key factors (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.35) for c in (BLUE, AQUA, GRAY, ORANGE)]
    ax.legend(handles, ["I: natural growth", "II, IV: selective growth",
                        "III, V, VII: hold and watch competitors",
                        "VI, VIII, IX: retrenchment (harvest / reduce)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=8.5)
    ax.set_title("Grand strategy from the IEFE analysis: region V,\n"
                 "'hold and watch competitors'", loc="left", fontsize=11)
    save(fig, "fig5_ie_matrix.png")


# 6. Stakeholder power / interest grid ----------------------------------------------
def fig_stakeholders():
    df = pd.read_csv(DATA / "stakeholders.csv")
    pos = {("High", "High"): (0.75, 0.75), ("High", "Low"): (0.25, 0.75),
           ("Low", "High"): (0.75, 0.25), ("Low", "Low"): (0.25, 0.25)}
    names = {("High", "High"): "D - key players", ("High", "Low"): "C - high power, low interest",
             ("Low", "High"): "B - high interest, low power", ("Low", "Low"): "A - low power, low interest"}
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for key, (x, y) in pos.items():
        col = BLUE if key == ("High", "High") else GRAY
        ax.add_patch(plt.Rectangle((x - 0.25, y - 0.25), 0.5, 0.5, facecolor=col, alpha=0.15,
                                   edgecolor=SURFACE, lw=3))
        ax.text(x - 0.23, y + 0.225, names[key], fontsize=9, color=INK2, va="top", weight="bold")
        items = df[(df["power"] == key[0]) & (df["interest"] == key[1])]["stakeholder"]
        ax.text(x, y - 0.03, "\n".join(wrap(i, 34) for i in items), ha="center", va="center",
                fontsize=9, color=INK, linespacing=1.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Level of interest  ->")
    ax.set_ylabel("Power  ->")
    ax.set_title("Stakeholder power / interest matrix", loc="left", fontsize=11)
    save(fig, "fig6_stakeholder_matrix.png")


if __name__ == "__main__":
    fig_degree_of_freedom()
    fig_life_cycle()
    fig_porter()
    fig_iefe_top()
    fig_ie_matrix()
    fig_stakeholders()
    print("Figures written to", FIG)
