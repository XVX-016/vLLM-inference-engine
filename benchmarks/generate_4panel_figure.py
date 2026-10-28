"""
Generate 4-panel prefix caching evaluation figure.

Panel A – Throughput vs lambda for off and share across individual seeds (P=1, budget=768T).
           Scatter points per seed (no connected means) showing plateaus at high load.

Panel B – First-admission prefix hit rate vs theoretical ceiling across P=1, 4, 16.
           Clean first-admission metric strictly <= theoretical ceiling.

Panel C – In-process prefill latency breakdown (measure_prefill_breakdown.py).
           Raw HF vs Engine copy vs Partial prefill, highlighting launch-boundedness.

Panel D – Empirical Plateau Summary Table across operating regimes.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = REPO_ROOT / "benchmarks" / "results" / "prefix_cache" / "prefix_cache_4panel.png"

def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    fig.suptitle('Prefix KV-Cache Sharing: Empirical Evaluation (Qwen2.5-0.5B, 768T Budget)', fontsize=13, fontweight='bold', y=0.98)

    # Panel A: Throughput vs Lambda (Individual seed points, no connected means)
    ax = axes[0, 0]
    lams = [3.0, 4.0, 6.0, 8.0]
    p1_off_42 = [38.9, 38.8, 64.3, np.nan]
    p1_off_43 = [39.1, 41.7, 71.8, 72.6]
    p1_share_42 = [65.6, 68.5, 82.5, 91.3]
    p1_share_43 = [56.5, 68.7, 91.5, 90.0]

    ax.scatter(lams, p1_off_42, color='#ef4444', marker='o', s=55, label='off (seed 42)')
    ax.scatter(lams, p1_off_43, color='#ef4444', marker='s', s=55, facecolors='none', edgecolors='#ef4444', linewidths=1.5, label='off (seed 43)')
    ax.scatter(lams, p1_share_42, color='#2563eb', marker='o', s=55, label='share (seed 42)')
    ax.scatter(lams, p1_share_43, color='#2563eb', marker='s', s=55, facecolors='none', edgecolors='#2563eb', linewidths=1.5, label='share (seed 43)')

    ax.set_title('A  Throughput vs Offered Load (P=1, Budget=768T)\n(Individual seed points; plateaus at ~90 vs ~72 tok/s)', fontsize=10, fontweight='bold')
    ax.set_xlabel(r'Offered Load $\lambda$ (req/s)', fontsize=9)
    ax.set_ylabel('Throughput (tok/s)', fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xticks(lams)
    ax.grid(True, linestyle='--', alpha=0.45)
    ax.legend(fontsize=8, loc='upper left')

    # Panel B: First-Admission Hit Rate vs Theoretical Ceiling
    ax = axes[0, 1]
    P_vals = ['P=1', 'P=4', 'P=16']
    x = np.arange(len(P_vals))
    width = 0.35

    theo = [0.561, 0.506, 0.506]
    measured_first = [0.558, 0.483, 0.040]

    ax.bar(x - width/2, theo, width, label='Theoretical Ceiling (Prompt blocks)', color='#94a3b8')
    ax.bar(x + width/2, measured_first, width, label='Measured (First Admission)', color='#2563eb')
    ax.set_title('B  First-Admission Prefix Hit Rate vs Pool Size P\n(Budget=768T; P=16 thrashes 768T pool)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(P_vals)
    ax.set_ylabel('Prefix Hit Rate', fontsize=9)
    ax.set_ylim(0, 0.70)
    ax.grid(True, linestyle='--', alpha=0.45, axis='y')
    ax.legend(fontsize=8, loc='upper right')
    for i, v in enumerate(measured_first):
        ax.text(x[i] + width/2, v + 0.015, f'{v*100:.1f}%', ha='center', fontsize=8, fontweight='bold')
    for i, v in enumerate(theo):
        ax.text(x[i] - width/2, v + 0.015, f'{v*100:.1f}%', ha='center', fontsize=8, color='#475569')

    # Panel C: Microbenchmark vs Serving Pre-fill Breakdown
    ax = axes[1, 0]
    labels = ['Raw HF\n(489 tok)', 'Engine off\n(489 tok)', 'Engine full\n(8 tok + 481 cached)']
    times = [55.43, 51.75, 58.06] # ms
    bars = ax.bar(labels, times, color=['#64748b', '#ef4444', '#10b981'], width=0.45)
    ax.set_title('C  In-Process Prefill Latency Breakdown\n(Kernel launch-bound; KV copy overhead ~0 ms)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=9)
    ax.set_ylim(0, 75)
    ax.grid(True, linestyle='--', alpha=0.45, axis='y')
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, t + 1.5, f'{t:.1f} ms', ha='center', fontsize=8.5, fontweight='bold')
    ax.annotate('Partial-prompt forward saves negligible time\non 0.5B model (58 ms vs 55 ms: launch-bound)', 
                xy=(2, 58.06), xytext=(0.85, 66), fontsize=8,
                arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'))

    # Panel D: Saturated Throughput Comparison Table / Range
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Operating Regime', '`off` Plateau', '`share` Plateau', 'Capacity Speedup', 'Primary Mechanism'],
        ['P=1 (S=64, 768T)', '64.3 – 72.6 tok/s', '82.5 – 91.5 tok/s', '1.24 – 1.28x', 'Memory frees KV slots'],
        ['P=4 (S=64, 768T)', '63.0 – 70.8 tok/s', '67.4 – 105.0 tok/s', '1.07 – 1.48x', 'Moderate sharing'],
        ['P=16 (768T, Thrash)', '38.7 – 41.1 tok/s', '41.6 – 42.6 tok/s', '1.03 – 1.10x', 'LRU cache thrashing'],
        ['P=16 (4096T, Resident)', '49.5 tok/s', '67.1 tok/s', '1.36x', 'Full prefix residency'],
        ['Control (P=200, 0% hit)', '40.2 – 41.1 tok/s', '42.5 – 42.7 tok/s', '1.00 – 1.05x', 'Zero runtime penalty'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.8)
    for i in range(5):
        table[(0, i)].set_facecolor('#f1f5f9')
        table[(0, i)].set_text_props(weight='bold')

    ax.set_title('D  Plateau Summary & Empirical Scaling Across Regimes', fontsize=10, fontweight='bold', pad=15)

    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUT_PNG}")

if __name__ == '__main__':
    main()
