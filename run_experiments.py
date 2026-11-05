"""
Run full simulation experiments for Energy-Aware Scheduling of AI Workloads in Kubernetes.
Evaluates 5 policies on the Alibaba PAI real dataset over 30 paired seeds.

Usage: python run_experiments.py
"""

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulator as sim

SEEDS = range(30)
LOADS = ["low", "medium", "high"]
POLICIES = ["fcfs", "default", "binpack", "energy", "meta_energy"]
LABEL = {
    "fcfs":        "FCFS (First-Fit)",
    "default":     "Default K8s",
    "binpack":     "Bin-packing",
    "energy":      "Energy-aware",
    "meta_energy": "Metadata-aware",
}
OUT_DIR = "figs"


def run_experiment_suite():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    def run_batch(tag, loads, sleep_enabled=True, wake=None, taint=False):
        w_delay = wake if wake is not None else sim.WAKE_DELAY
        print(f"\n--- Running experiment: {tag} (loads={loads}, sleep={sleep_enabled}, wake={w_delay}m, taint={taint}) ---")
        for lvl in loads:
            print(f"  Load level: {lvl} ...", flush=True)
            for s in SEEDS:
                jobs = sim.sample_workload(s, lvl)
                for p in POLICIES:
                    res = sim.simulate(
                        p, jobs, seed=s,
                        sleep_enabled=sleep_enabled,
                        taint=taint,
                        wake_delay=w_delay
                    )
                    res.update(exp=tag, load=lvl, seed=s)
                    rows.append(res)

    # 1. Main experiment
    run_batch("main", LOADS, sleep_enabled=True)

    # 2. Ablation: No sleep / power-down disabled
    run_batch("nosleep", LOADS, sleep_enabled=False)

    # 3. Robustness: CPU-only pods forbidden on GPU nodes (taint)
    run_batch("taint", LOADS, sleep_enabled=True, taint=True)

    # 4. Sensitivity: Wake-up delays of 2, 5, 10 min at medium load
    for w in (2, 5, 10):
        run_batch(f"wake{w}", ["medium"], sleep_enabled=True, wake=float(w))

    df = pd.DataFrame(rows)
    df.to_csv("results_raw.csv", index=False)
    print(f"\nSaved raw results ({len(df)} rows) to results_raw.csv")
    return df


def ci95(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def compute_summary(df):
    summary = []
    for exp in df["exp"].unique():
        exp_df = df[df["exp"] == exp]
        for load in exp_df["load"].unique():
            load_df = exp_df[exp_df["load"] == load]
            def_energy = load_df[load_df["policy"] == "default"].sort_values("seed")["energy_kwh"].values

            for pol in POLICIES:
                d = load_df[load_df["policy"] == pol].sort_values("seed")
                if len(d) == 0:
                    continue
                pol_energy = d["energy_kwh"].values

                # Paired savings vs Default
                sav = (def_energy - pol_energy) / def_energy * 100.0
                if pol != "default":
                    _, pval = stats.ttest_rel(def_energy, pol_energy)
                else:
                    pval = np.nan

                rec = dict(
                    exp=exp,
                    load=load,
                    policy=pol,
                    n=len(d),
                    energy=float(d["energy_kwh"].mean()),
                    energy_ci=ci95(d["energy_kwh"]),
                    saving=float(sav.mean()),
                    saving_ci=ci95(sav),
                    p_energy=float(pval) if not np.isnan(pval) else np.nan,
                    active=float(d["mean_active_nodes"].mean()),
                    active_ci=ci95(d["mean_active_nodes"]),
                    cpu_util=float(d["mean_cpu_util_active"].mean() * 100.0),
                    gpu_util=float(d["mean_gpu_util_active"].mean() * 100.0),
                    wait=float(d["mean_wait"].mean()),
                    wait_ci=ci95(d["mean_wait"]),
                    p95=float(d["p95_wait"].mean()),
                    p95_ci=ci95(d["p95_wait"]),
                    jct=float(d["mean_jct"].mean()),
                    slow=float(d["mean_slowdown"].mean()),
                    wakes=float(d["wakes"].mean()),
                    completed=float((d["completed"] / d["jobs"]).mean() * 100.0),
                )
                summary.append(rec)

    sdf = pd.DataFrame(summary)
    sdf.to_csv("results_summary.csv", index=False)
    print("Saved summary table to results_summary.csv")
    return sdf


def fmt_p(p):
    if np.isnan(p):
        return "--"
    return "$<0.001$" if p < 0.001 else f"{p:.3f}"


def export_latex_tables(S):
    fix = lambda t: re.sub(r"(?<![\$\-\w])-(?=\d)", "$-$", t)

    # 1. Main Energy Table
    lines_main = []
    for load in LOADS:
        for pol in POLICIES:
            r = S[(S["exp"] == "main") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            sav_str = "--" if pol == "default" else f"{r.saving:.1f} $\\pm$ {r.saving_ci:.1f}"
            lbl = load.capitalize() if pol == POLICIES[0] else ""
            lines_main.append(
                f"{lbl} & {LABEL[pol]} & {r.energy:.1f} $\\pm$ {r.energy_ci:.1f} & "
                f"{sav_str} & {fmt_p(r.p_energy)} & {r.active:.2f} & {r.gpu_util:.1f} \\\\"
            )
        if load != LOADS[-1]:
            lines_main.append("\\midrule")
    with open("tab_main.tex", "w") as f:
        f.write(fix("\n".join(lines_main)))

    # 2. Performance Table (Wait, P95, Slowdown, Wakes, Done)
    lines_perf = []
    for load in LOADS:
        for pol in POLICIES:
            r = S[(S["exp"] == "main") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            lbl = load.capitalize() if pol == POLICIES[0] else ""
            lines_perf.append(
                f"{lbl} & {LABEL[pol]} & {r.wait:.2f} $\\pm$ {r.wait_ci:.2f} & "
                f"{r.p95:.1f} & {r.slow:.3f} & {r.wakes:.1f} & {r.completed:.1f} \\\\"
            )
        if load != LOADS[-1]:
            lines_perf.append("\\midrule")
    with open("tab_perf.tex", "w") as f:
        f.write(fix("\n".join(lines_perf)))

    # 3. Ablation Table (With vs Without power-down)
    lines_abl = []
    for load in LOADS:
        for pol in POLICIES:
            a = S[(S["exp"] == "main") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            b = S[(S["exp"] == "nosleep") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            lbl = load.capitalize() if pol == POLICIES[0] else ""
            lines_abl.append(
                f"{lbl} & {LABEL[pol]} & {b.energy:.1f} & {b.saving:.1f} & "
                f"{a.energy:.1f} & {a.saving:.1f} \\\\"
            )
        if load != LOADS[-1]:
            lines_abl.append("\\midrule")
    with open("tab_ablation.tex", "w") as f:
        f.write(fix("\n".join(lines_abl)))

    # 4. Taint Table (CPU-only pod isolation)
    lines_taint = []
    for load in LOADS:
        for pol in POLICIES:
            a = S[(S["exp"] == "main") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            b = S[(S["exp"] == "taint") & (S["load"] == load) & (S["policy"] == pol)].iloc[0]
            lbl = load.capitalize() if pol == POLICIES[0] else ""
            lines_taint.append(
                f"{lbl} & {LABEL[pol]} & {a.energy:.1f} & {a.saving:.1f} & "
                f"{b.energy:.1f} & {b.saving:.1f} & {b.wait:.2f} \\\\"
            )
        if load != LOADS[-1]:
            lines_taint.append("\\midrule")
    with open("tab_taint.tex", "w") as f:
        f.write(fix("\n".join(lines_taint)))

    # 5. Wake Delay Sensitivity Table
    lines_wake = []
    for w in (2, 5, 10):
        for pol in POLICIES:
            r = S[(S["exp"] == f"wake{w}") & (S["policy"] == pol)].iloc[0]
            lbl = str(w) if pol == POLICIES[0] else ""
            lines_wake.append(
                f"{lbl} & {LABEL[pol]} & {r.energy:.1f} & {r.saving:.1f} & "
                f"{r.wait:.2f} & {r.p95:.1f} \\\\"
            )
        if w != 10:
            lines_wake.append("\\midrule")
    with open("tab_wake.tex", "w") as f:
        f.write(fix("\n".join(lines_wake)))

    print("Exported all LaTeX table fragments successfully.")


def generate_figures(S):
    plt.rcParams.update({"font.size": 8.5, "font.family": "serif"})
    colors = {
        "fcfs":        "#b07aa1",
        "default":     "#8c8c8c",
        "binpack":     "#5b8dd6",
        "energy":      "#e67e22",
        "meta_energy": "#2e8b57",
    }
    loads = ["low", "medium", "high"]
    x = np.arange(3)
    n_pols = len(POLICIES)
    w = 0.16

    # Figure 1: Total cluster energy bar chart
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for k, pol in enumerate(POLICIES):
        v = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["energy"].iloc[0] for l in loads]
        e = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["energy_ci"].iloc[0] for l in loads]
        pos = x + (k - (n_pols - 1) / 2.0) * w
        ax.bar(pos, v, w, yerr=e, capsize=2, label=LABEL[pol], color=colors[pol], edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low\n(~120 pods)", "Medium\n(~400 pods)", "High\n(~550 pods)"])
    ax.set_ylabel("Total energy (kWh / 54 h)")
    ax.legend(frameon=True, fontsize=6.5, loc="upper left", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_energy.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_energy.png"), dpi=300)
    plt.close(fig)

    # Figure 2: Energy savings relative to default K8s
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    for pol in ["fcfs", "binpack", "energy", "meta_energy"]:
        v = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["saving"].iloc[0] for l in loads]
        e = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["saving_ci"].iloc[0] for l in loads]
        ax.errorbar(x, v, yerr=e, marker="o", capsize=3, label=LABEL[pol], color=colors[pol], linewidth=1.2)
    ax.axhline(0, color="black", lw=0.6, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.set_xlabel("Offered load level")
    ax.set_ylabel("Energy saving vs default (%)")
    ax.legend(frameon=True, fontsize=6.5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig_saving.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_saving.png"), dpi=300)
    plt.close(fig)

    # Figure 3: Active nodes and P95 waiting time
    fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.6))
    for k, pol in enumerate(POLICIES):
        pos = x + (k - (n_pols - 1) / 2.0) * w
        v_act = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["active"].iloc[0] for l in loads]
        axs[0].bar(pos, v_act, w, color=colors[pol], label=LABEL[pol], edgecolor="black", linewidth=0.3)
        v_p95 = [S[(S["exp"] == "main") & (S["load"] == l) & (S["policy"] == pol)]["p95"].iloc[0] for l in loads]
        axs[1].bar(pos, v_p95, w, color=colors[pol], edgecolor="black", linewidth=0.3)

    axs[0].set_xticks(x); axs[0].set_xticklabels(["L", "M", "H"]); axs[0].set_title("Mean active nodes", fontsize=8)
    axs[1].set_xticks(x); axs[1].set_xticklabels(["L", "M", "H"]); axs[1].set_title("95th pct. wait (min)", fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT_DIR, "fig_nodes_wait.pdf"))
    fig.savefig(os.path.join(OUT_DIR, "fig_nodes_wait.png"), dpi=300)
    plt.close(fig)

    print("Generated all publication-grade figures successfully.")


def main():
    print("=" * 65)
    print("ENERGY-AWARE SCHEDULING SIMULATION (ALIBABA PAI REAL TRACE)")
    print("=" * 65)
    df = run_experiment_suite()
    S = compute_summary(df)
    export_latex_tables(S)
    generate_figures(S)
    print("\nALL SIMULATION EXPERIMENTS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
