from pathlib import Path
import matplotlib.pyplot as plt
from evidence import load_design_table, plot_evidence_1d

MARGE_CSV = "design/ghc_HIGHZ_NGC_marge.csv"
FEATURES = ["omega_m", "w0", "ln1e10As"]


def main():
    table = load_design_table(MARGE_CSV)
    Path("design").mkdir(exist_ok=True)
    Path("evidence").mkdir(exist_ok=True)
    table.to_csv("design/design_HIGHZ_NGC.csv", index=False)

    fig, axes = plt.subplots(1, len(FEATURES), figsize=(5 * len(FEATURES), 4.5))
    for ax, feature in zip(axes, FEATURES):
        plot_evidence_1d(table, feature, ax=ax)
    fig.tight_layout()
    fig.savefig("evidence/evidence_trends_HIGHZ_NGC.png", dpi=150)
    return table


if __name__ == "__main__":
    main()