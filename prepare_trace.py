"""
Preprocess the Alibaba PAI cluster trace for energy-aware scheduling simulation.

Reads raw CSVs from the Alibaba PAI GPU cluster trace (Alibaba cluster-trace-gpu-v2020 / Weng et al., NSDI 2022),
selects a representative 48-hour window, converts tasks to simulation pods, and
saves the processed trace to CSV.

Usage:  python prepare_trace.py
"""
import os, sys
import numpy as np
import pandas as pd

DATA_DIR   = r"c:\Users\haris\Downloads\pai_sample"
OUT_DIR    = "trace"
WIN_START  = 24.0   # hours from trace start (skip first day for warm-up)
WIN_HOURS  = 48.0
MAX_INST   = 50     # cap instances per task to avoid pathological cases

# --------------- task-type classification ---------------
JOB_TYPE = {
    "tensorflow": "training", "worker": "training", "PyTorchWorker": "training",
    "MWorker": "training", "OpenmpiWorker": "training", "chief": "training",
    "ps": "param_server", "OpenmpiTracker": "param_server",
    "xComputeWorker": "preprocessing", "ReduceTask": "preprocessing",
    "DecoderWorker": "preprocessing", "OssToVolumeWorker": "preprocessing",
    "evaluator": "evaluation", "TensorboardTask": "evaluation",
    "JupyterTask": "evaluation", "BladeMain": "other",
}
# default utilisation fractions when sensor data is missing
DEFAULT_KAPPA = {"training": 0.60, "param_server": 0.30, "preprocessing": 0.80,
                 "evaluation": 0.40, "other": 0.50}
DEFAULT_GAMMA = {"training": 0.70, "param_server": 0.0, "preprocessing": 0.30,
                 "evaluation": 0.20, "other": 0.30}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- load ----
    print("Loading raw CSVs …")
    jobs    = pd.read_csv(os.path.join(DATA_DIR, "job_sample.csv"))
    tasks   = pd.read_csv(os.path.join(DATA_DIR, "task_sample.csv"))
    sensors = pd.read_csv(os.path.join(DATA_DIR, "sensor_by_task.csv"))

    # ---- filter ----
    jobs  = jobs[jobs["status"] == "Terminated"].dropna(subset=["start_time", "end_time"])
    tasks = tasks[tasks["status"] == "Terminated"].dropna(subset=["start_time", "end_time"])
    tasks = tasks[tasks["end_time"] > tasks["start_time"]]
    tasks = tasks.dropna(subset=["plan_cpu", "plan_mem"])
    print(f"  Terminated jobs: {len(jobs):,}   tasks: {len(tasks):,}")

    # ---- normalise times to trace start ----
    t0 = jobs["start_time"].min()
    jobs  = jobs.copy();  jobs["start_time"]  -= t0; jobs["end_time"]  -= t0
    tasks = tasks.copy(); tasks["start_time"] -= t0; tasks["end_time"] -= t0

    # ---- 48-h window ----
    ws = WIN_START * 3600
    we = ws + WIN_HOURS * 3600
    print(f"Window: {WIN_START:.0f} h – {WIN_START + WIN_HOURS:.0f} h")

    win_jobs  = jobs[(jobs["start_time"] >= ws) & (jobs["start_time"] < we)]
    win_tasks = tasks[tasks["job_name"].isin(win_jobs["job_name"])]
    win_tasks = win_tasks[(win_tasks["start_time"] >= ws) & (win_tasks["start_time"] < we)]
    print(f"  Jobs in window: {len(win_jobs):,}   Tasks: {len(win_tasks):,}")

    # ---- merge sensor data ----
    win_tasks = win_tasks.merge(
        sensors[["job_name", "task_name", "cpu_usage", "gpu_wrk_util", "n_inst"]]
        .rename(columns={"n_inst": "sensor_n"}),
        on=["job_name", "task_name"], how="left")

    # ---- expand to pods ----
    pods = []
    pid = 0
    for _, t in win_tasks.iterrows():
        n_inst = min(int(t["inst_num"]) if pd.notna(t["inst_num"]) else 1, MAX_INST)
        cpu    = t["plan_cpu"] / 100.0          # centcores → cores
        mem    = t["plan_mem"]                   # GB
        gpu    = t["plan_gpu"] / 100.0 if pd.notna(t["plan_gpu"]) else 0.0  # pct → GPUs
        dur    = (t["end_time"] - t["start_time"]) / 60.0   # seconds → minutes
        arr    = (t["start_time"] - ws) / 60.0               # relative to window start

        gtype  = t["gpu_type"] if pd.notna(t["gpu_type"]) and gpu > 0 else "none"
        jtype  = JOB_TYPE.get(t["task_name"], "other")

        sn = t["sensor_n"] if pd.notna(t["sensor_n"]) else n_inst
        if pd.notna(t["cpu_usage"]) and t["plan_cpu"] > 0 and sn > 0:
            kappa = min(t["cpu_usage"] / (sn * t["plan_cpu"]), 1.0)
        else:
            kappa = DEFAULT_KAPPA.get(jtype, 0.5)
        if pd.notna(t["gpu_wrk_util"]) and gpu > 0 and sn > 0:
            gamma = min(t["gpu_wrk_util"] / (sn * 100.0), 1.0)
        else:
            gamma = 0.0 if gpu == 0 else DEFAULT_GAMMA.get(jtype, 0.3)

        if cpu <= 0 or mem <= 0 or dur <= 0 or arr < 0:
            continue

        for _ in range(n_inst):
            pods.append(dict(
                pod_id=pid, arrival_min=round(arr, 2), duration_min=round(max(dur, 1.0), 2),
                cpu=round(cpu, 2), mem=round(mem, 2), gpu=round(gpu, 2),
                gpu_type=gtype, kappa=round(kappa, 4), gamma=round(gamma, 4),
                job_type=jtype, job_name=t["job_name"], task_name=t["task_name"]))
            pid += 1

    df = pd.DataFrame(pods)
    out = os.path.join(OUT_DIR, "pods.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved {len(df):,} pods to {out}")

    # ---- statistics ----
    print(f"\n{'='*60}")
    print("POD  TRACE  STATISTICS")
    print(f"{'='*60}")
    print(f"Total pods:   {len(df):,}")
    print(f"Unique jobs:  {df['job_name'].nunique():,}")
    print(f"\nJob-type distribution:")
    print(df["job_type"].value_counts().to_string())
    print(f"\nGPU-type distribution (GPU pods):")
    print(df[df["gpu"] > 0]["gpu_type"].value_counts().to_string())
    print(f"\nResource-request statistics:")
    for c in ["cpu", "mem", "gpu", "duration_min", "kappa", "gamma"]:
        s = df[c]
        print(f"  {c:15s}  mean={s.mean():8.2f}  med={s.median():8.2f}  max={s.max():10.2f}")
    n_gpu = (df["gpu"] > 0).sum()
    print(f"\nPods needing GPUs: {n_gpu:,} ({n_gpu/len(df)*100:.1f}%)")

    tot_cpu_min = (df["cpu"] * df["duration_min"]).sum()
    tot_gpu_min = (df["gpu"] * df["duration_min"]).sum()
    win_min = WIN_HOURS * 60
    print(f"\nAverage concurrent demand over {WIN_HOURS:.0f} h:")
    print(f"  CPU cores: {tot_cpu_min / win_min:.0f}")
    print(f"  GPUs:      {tot_gpu_min / win_min:.1f}")

    print(f"\nMedian duration by job type (minutes):")
    for jt in sorted(df["job_type"].unique()):
        print(f"  {jt:20s}  {df[df['job_type']==jt]['duration_min'].median():.1f}")

    # ---- compute suggested sampling fractions ----
    # Cluster: 20 nodes, 1664 cores, 74 GPUs
    CLUSTER_CPU, CLUSTER_GPU = 1664, 74
    target_utils = {"low": 0.25, "medium": 0.50, "high": 0.75}
    avg_cpu = tot_cpu_min / win_min  # at 100% sampling
    print(f"\nSuggested sampling fractions (cluster: {CLUSTER_CPU} cores, {CLUSTER_GPU} GPUs):")
    for lvl, u in target_utils.items():
        frac = CLUSTER_CPU * u / avg_cpu if avg_cpu > 0 else 0.01
        print(f"  {lvl:8s}  target util {u:.0%}  →  fraction {frac:.4f}  ({frac*len(df):.0f} pods)")


if __name__ == "__main__":
    main()
