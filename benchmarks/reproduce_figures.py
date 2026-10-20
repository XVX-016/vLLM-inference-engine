"""
Reproduce and regenerate the consolidated evaluation figures from the validated experimental run:
- Generates: results/serving_evaluation_consolidated.png
- Reads data from: results/final_all_reqs_results.json
"""
import collections
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_JSON = RESULTS_DIR / "final_all_reqs_results.json"
OUT_PNG = RESULTS_DIR / "serving_evaluation_consolidated.png"


def main():
    if not RESULTS_JSON.exists():
        raise FileNotFoundError(f"Missing master results file: {RESULTS_JSON}")

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Combine runs
    req1 = data.get("requirement_1_naive_validation", [])
    req4 = data.get("requirement_4_high_load_grid", [])
    req5 = data.get("requirement_5_ablation", [])
    for r in req5:
        r["system"] = "paged_bs3"

    all_runs = req1 + req4 + req5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    ax1, ax2 = axes[0], axes[1]

    data_by_sys = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in all_runs:
        data_by_sys[r["system"]][r["lam"]].append(r)

    styles = {
        "naive": {"color": "#dc2626", "label": "Naive Static-Cache (Slots=3)", "marker": "s"},
        "paged": {"color": "#2563eb", "label": "Paged Continuous (Unbounded)", "marker": "o"},
        "paged_bs3": {"color": "#16a34a", "label": "Paged Ablation (max_bs=3)", "marker": "^"},
    }

    for sys_name, lams_dict in data_by_sys.items():
        st = styles.get(sys_name, {"color": "gray", "label": sys_name, "marker": "d"})
        sorted_lams = sorted(lams_dict.keys())

        offered_means = [np.mean([x["offered_load_tok_s"] for x in lams_dict[l]]) for l in sorted_lams]
        goodput_means = [np.mean([x["goodput_tok_s"] for x in lams_dict[l]]) for l in sorted_lams]
        ttft_p99_means = [np.mean([x["ttft_p99"] for x in lams_dict[l] if x["ttft_p99"] is not None] or [np.nan]) for l in sorted_lams]

        ax1.plot(offered_means, goodput_means, color=st["color"], linestyle="-", linewidth=2, label=st["label"])

        for l in sorted_lams:
            pts = lams_dict[l]
            off_m = np.mean([x["offered_load_tok_s"] for x in pts])
            gp_m = np.mean([x["goodput_tok_s"] for x in pts])
            censored = any(x["is_timeout_censored"] for x in pts)

            face_col = "white" if censored else st["color"]
            ax1.scatter([off_m], [gp_m], color=st["color"], marker=st["marker"], s=90, facecolors=face_col, edgecolors=st["color"], linewidth=2, zorder=5)

            ttft_valid = [x["ttft_p99"] for x in pts if x["ttft_p99"] is not None]
            if ttft_valid:
                ttft_m = np.mean(ttft_valid)
                ax2.scatter([l], [ttft_m], color=st["color"], marker=st["marker"], s=90, facecolors=face_col, edgecolors=st["color"], linewidth=2, zorder=5)

        valid_l = [l for l, t in zip(sorted_lams, ttft_p99_means) if not np.isnan(t)]
        valid_t = [t for t in ttft_p99_means if not np.isnan(t)]
        if valid_l:
            ax2.plot(valid_l, valid_t, color=st["color"], linestyle="-", linewidth=2, label=st["label"])

    ax1.set_title("Goodput vs. Offered Load\n(Hollow = Timeout Censored)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Offered Load (tokens/s)", fontsize=11)
    ax1.set_ylabel("Goodput (tokens/s)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.9)

    ax2.set_title("TTFT P99 vs. Arrival Rate $\\lambda$\n(Hollow = Timeout Censored)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Arrival Rate $\\lambda$ (req/s)", fontsize=11)
    ax2.set_ylabel("TTFT P99 (s)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Successfully generated plot -> {OUT_PNG}")


if __name__ == "__main__":
    main()
