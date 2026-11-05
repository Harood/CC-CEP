"""
Energy-Aware Scheduling of AI Workloads in Kubernetes
Discrete-time cluster simulator using the Alibaba PAI cluster trace dataset.

This simulator models a heterogeneous Kubernetes cluster whose node configurations
and workload requests are derived from the Alibaba PAI GPU cluster trace
(Alibaba cluster-trace-gpu-v2020 / Weng et al., NSDI 2022).

Policies evaluated:
  fcfs        : First-Come First-Served / First-Fit baseline (no scoring)
  default     : Kubernetes default scoring (NodeResourcesFit LeastAllocated + BalancedAllocation)
  binpack     : Power-agnostic consolidation (MostAllocated scoring)
  energy      : Proposed marginal power-aware consolidation policy
  meta_energy : Metadata-aware energy policy (workload type, duration affinity, accelerator efficiency)
"""

import os
import numpy as np
import pandas as pd

STEP = 1.0                # minutes per simulation step
ARRIVAL_WINDOW = 48 * 60  # 48-hour arrival window (minutes)
T_END = 54 * 60           # 54-hour evaluation window (48 h arrivals + 6 h drain)
IDLE_TIMEOUT = 10.0       # minutes an empty node waits before entering sleep state
WAKE_DELAY = 2.0          # minutes required to wake a sleeping node

# Node configurations derived directly from Alibaba PAI machine specifications (machine_spec.csv):
# (type_name, count, cpu_cores, mem_gb, gpus, gpu_arch, p_idle(W), p_cpu_dyn(W), p_gpu_dyn(W), p_sleep(W))
NODE_SPECS = [
    ("cpu-std",   2, 96, 512, 0, "none", 150.0, 250.0,    0.0, 15.0),
    ("gpu-p100",  4, 64, 512, 2, "P100", 250.0, 220.0,  500.0, 20.0),
    ("gpu-t4",    8, 96, 512, 2, "T4",   180.0, 250.0,  140.0, 18.0),
    ("gpu-v100",  2, 96, 512, 8, "V100", 450.0, 250.0, 2000.0, 35.0),
]

# Total cluster capacity:
# 16 nodes: 2*96 + 4*64 + 8*96 + 2*96 = 1,408 CPU cores, 8,192 GB RAM, 40 GPUs (8 P100, 16 T4, 16 V100)

# Number of unique jobs sampled per 48h trace to produce Low, Medium, and High cluster load
JOB_SAMPLE_SIZES = {
    "low":    40,   # ~100-150 pods, ~20-25% active cluster utilization
    "medium": 80,   # ~350-450 pods, ~45-55% active cluster utilization
    "high":   130,  # ~500-650 pods, ~70-80% active cluster utilization
}


class Node:
    __slots__ = (
        "idx", "type", "cpu", "mem", "gpu", "gpu_arch",
        "p_idle", "p_cpu", "p_gpu", "p_sleep",
        "used_cpu", "used_mem", "used_gpu",
        "state", "ready_at", "idle_since", "n_jobs", "running"
    )

    def __init__(self, idx, spec):
        (self.type, _, self.cpu, self.mem, self.gpu, self.gpu_arch,
         self.p_idle, self.p_cpu, self.p_gpu, self.p_sleep) = spec
        self.idx = idx
        self.used_cpu = 0.0
        self.used_mem = 0.0
        self.used_gpu = 0.0
        self.state = "active"          # "active" | "sleep"
        self.ready_at = 0.0
        self.idle_since = 0.0
        self.n_jobs = 0                # count of reserved or running pods
        self.running = []              # list of running pod dicts

    def fits(self, j):
        if self.cpu - self.used_cpu < j["cpu"] - 1e-6:
            return False
        if self.mem - self.used_mem < j["mem"] - 1e-6:
            return False
        if self.gpu - self.used_gpu < j["gpu"] - 1e-6:
            return False
        if j["gpu"] > 0 and self.gpu == 0:
            return False
        return True

    def marginal_power(self, j):
        w = (j["cpu"] * j["kappa"] / self.cpu) * self.p_cpu
        if self.gpu > 0 and j["gpu"] > 0:
            w += (j["gpu"] * j["gamma"] / self.gpu) * self.p_gpu
        return w

    def most_allocated(self, j):
        c = (self.used_cpu + j["cpu"]) / self.cpu
        m = (self.used_mem + j["mem"]) / self.mem
        g = (self.used_gpu + j["gpu"]) / self.gpu if self.gpu > 0 else 0.0
        return (c + m + g) / (3.0 if self.gpu > 0 else 2.0)

    def least_allocated(self, j):
        return 1.0 - self.most_allocated(j)

    def balanced(self, j):
        c = (self.used_cpu + j["cpu"]) / self.cpu
        m = (self.used_mem + j["mem"]) / self.mem
        return 1.0 - abs(c - m)

    def power(self, t):
        if self.state == "sleep":
            return self.p_sleep
        if t < self.ready_at:
            return self.p_idle
        cpu_u = sum(j["cpu"] * j["kappa"] for j in self.running) / self.cpu if self.cpu > 0 else 0.0
        p = self.p_idle + self.p_cpu * cpu_u
        if self.gpu > 0:
            gpu_u = sum(j["gpu"] * j["gamma"] for j in self.running) / self.gpu
            p += self.p_gpu * gpu_u
        return p


def make_cluster():
    nodes, idx = [], 0
    for spec in NODE_SPECS:
        for _ in range(spec[1]):
            nodes.append(Node(idx, spec))
            idx += 1
    return nodes


# Global cache for trace pods
_GLOBAL_PODS = None

def get_trace_pods(path=None):
    global _GLOBAL_PODS
    if _GLOBAL_PODS is None:
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "trace", "pods.csv")
        _GLOBAL_PODS = pd.read_csv(path)
    return _GLOBAL_PODS


def sample_workload(seed, load_level, pods_df=None):
    """
    Sample a coherent 48-hour workload trace from the Alibaba PAI dataset.
    Sampling by unique job preserves multi-task structures (workers + parameter servers),
    while scheduling each task instance as an individual pod (ignoring gang scheduling).
    """
    if pods_df is None:
        pods_df = get_trace_pods()

    unique_jobs = list(dict.fromkeys(pods_df["job_name"]))
    n_sample = JOB_SAMPLE_SIZES[load_level]

    rng = np.random.default_rng(seed)
    chosen_jobs = set(rng.choice(unique_jobs, size=n_sample, replace=False))

    sub = pods_df[pods_df["job_name"].isin(chosen_jobs)].copy()
    sub.sort_values(by=["arrival_min", "pod_id"], inplace=True)

    jobs = []
    for k, row in enumerate(sub.to_dict(orient="records")):
        row["id"] = k
        row["arrival"] = float(row["arrival_min"])
        row["dur"] = float(row["duration_min"])
        row["start"] = None
        row["end"] = None
        row["node"] = None
        jobs.append(row)
    return jobs


def choose_active(policy, cands, j):
    if policy == "fcfs":
        # First-Fit among active feasible nodes (lowest node index)
        return cands[0]

    if policy == "default":
        # Kubernetes default scoring: LeastAllocated + BalancedAllocation
        return max(cands, key=lambda n: (n.least_allocated(j) + n.balanced(j), -n.idx))

    if policy == "binpack":
        # Power-agnostic bin-packing: MostAllocated
        return max(cands, key=lambda n: (n.most_allocated(j), -n.idx))

    if policy == "energy":
        # Proposed marginal dynamic power minimization, tie-broken by MostAllocated
        return min(cands, key=lambda n: (round(n.marginal_power(j), 3),
                                         -n.most_allocated(j),
                                         n.idx))

    if policy == "meta_energy":
        # Metadata-aware energy policy:
        # 1. CPU-only pods strictly prioritize CPU-only nodes
        # 2. For GPU pods, prefer energy-efficient accelerator architecture (T4)
        # 3. Co-locate pods with similar runtimes to minimize idle stranding
        cpu_cands = [n for n in cands if n.gpu == 0]
        if j["gpu"] == 0 and cpu_cands:
            return min(cpu_cands, key=lambda n: (round(n.marginal_power(j), 3),
                                                 -n.most_allocated(j),
                                                 n.idx))

        def meta_score(n):
            mp = n.marginal_power(j)
            penalty = 0.0
            if j["gpu"] == 0 and n.gpu > 0:
                penalty += 120.0
            if j["gpu"] > 0:
                if n.gpu_arch == "T4":
                    penalty -= 35.0
                elif n.gpu_arch == "V100":
                    penalty += 35.0
            if n.running:
                avg_dur = sum(r["dur"] for r in n.running) / len(n.running)
                dur_diff = abs(np.log1p(j["dur"]) - np.log1p(avg_dur))
            else:
                dur_diff = 0.0
            return (round(mp + penalty, 3), dur_diff, -n.most_allocated(j), n.idx)

        return min(cands, key=meta_score)

    raise ValueError(f"Unknown policy: {policy}")


def choose_sleeping(policy, cands, j, rng):
    if policy == "fcfs":
        return cands[0]

    if policy == "default":
        # Simulates power-unaware autoscaler: arbitrary sleeping node
        return cands[int(rng.integers(len(cands)))]

    if policy == "binpack":
        return min(cands, key=lambda n: n.idx)

    if policy == "energy":
        # Minimum idle power + marginal power
        return min(cands, key=lambda n: (n.p_idle + n.marginal_power(j), n.idx))

    if policy == "meta_energy":
        # If CPU-only job, wake CPU-only node
        cpu_cands = [n for n in cands if n.gpu == 0]
        if j["gpu"] == 0 and cpu_cands:
            return min(cpu_cands, key=lambda n: (n.p_idle + n.marginal_power(j), n.idx))

        def sleep_score(n):
            base = n.p_idle + n.marginal_power(j)
            if j["dur"] < 30.0 and n.gpu_arch == "V100":
                base += 300.0   # Avoid waking 450W V100 node for ephemeral job
            if j["gpu"] == 0 and n.gpu > 0:
                base += 500.0   # Avoid waking GPU node for CPU pod
            if j["gpu"] > 0 and n.gpu_arch == "T4":
                base -= 50.0    # Prioritize waking energy-efficient T4 accelerator
            return (base, n.idx)

        return min(cands, key=sleep_score)

    raise ValueError(f"Unknown policy: {policy}")


def simulate(policy, jobs_in, seed=0, sleep_enabled=True, taint=False, wake_delay=WAKE_DELAY):
    rng = np.random.default_rng(10_000 + seed)
    jobs = [dict(j) for j in jobs_in]
    nodes = make_cluster()
    queue = []
    pending_start = []
    next_arrival = 0
    energy_wh = 0.0
    active_node_sum = 0
    cpu_util_sum = 0.0
    gpu_util_sum = 0.0
    wakes = 0
    steps = int(T_END / STEP)

    # Optional taint restriction: CPU-only pods may not run on GPU nodes
    def pod_ok(n, j):
        if taint and j["gpu"] == 0 and n.gpu > 0:
            return False
        return True

    for s in range(steps):
        t = s * STEP

        # 1. Job completions
        for n in nodes:
            if n.running:
                done = [j for j in n.running if j["end"] <= t]
                if done:
                    for j in done:
                        n.running.remove(j)
                        n.used_cpu -= j["cpu"]
                        n.used_mem -= j["mem"]
                        n.used_gpu -= j["gpu"]
                        n.n_jobs -= 1
                    if n.n_jobs == 0:
                        n.idle_since = t

        # 2. Woken nodes become ready: pending pods start execution
        if pending_start:
            still = []
            for j in pending_start:
                if j["start"] <= t:
                    nodes[j["node"]].running.append(j)
                else:
                    still.append(j)
            pending_start = still

        # 3. Node power-down: empty nodes sleep after IDLE_TIMEOUT minutes
        if sleep_enabled:
            for n in nodes:
                if (n.state == "active" and n.n_jobs == 0 and
                        t - n.idle_since >= IDLE_TIMEOUT and t >= n.ready_at):
                    n.state = "sleep"

        # 4. Job arrivals
        while next_arrival < len(jobs) and jobs[next_arrival]["arrival"] <= t:
            queue.append(jobs[next_arrival])
            next_arrival += 1

        # 5. Scheduling pass
        if queue:
            remaining = []
            for j in queue:
                act = [n for n in nodes if n.state == "active" and n.fits(j) and pod_ok(n, j)]
                if act:
                    n = choose_active(policy, act, j)
                else:
                    slp = [n for n in nodes if n.state == "sleep" and n.fits(j) and pod_ok(n, j)]
                    if not slp:
                        remaining.append(j)
                        continue
                    n = choose_sleeping(policy, slp, j, rng)
                    n.state = "active"
                    n.ready_at = t + wake_delay
                    n.idle_since = t
                    wakes += 1

                n.used_cpu += j["cpu"]
                n.used_mem += j["mem"]
                n.used_gpu += j["gpu"]
                n.n_jobs += 1
                j["node"] = n.idx
                j["start"] = max(t, n.ready_at)
                j["end"] = j["start"] + j["dur"]

                if j["start"] <= t:
                    n.running.append(j)
                else:
                    pending_start.append(j)

            queue = remaining

        # 6. Accounting
        act_nodes = [n for n in nodes if n.state == "active"]
        active_node_sum += len(act_nodes)

        # Fast power accounting
        for n in nodes:
            if n.state == "sleep":
                energy_wh += n.p_sleep * (STEP / 60.0)
            elif t < n.ready_at:
                energy_wh += n.p_idle * (STEP / 60.0)
            else:
                p = n.p_idle
                if n.running:
                    p += n.p_cpu * (sum(j["cpu"] * j["kappa"] for j in n.running) / n.cpu)
                    if n.gpu > 0:
                        p += n.p_gpu * (sum(j["gpu"] * j["gamma"] for j in n.running) / n.gpu)
                energy_wh += p * (STEP / 60.0)

        if act_nodes:
            c_u = [sum(j["cpu"] * j["kappa"] for j in n.running) / n.cpu for n in act_nodes]
            cpu_util_sum += sum(c_u) / len(c_u)

            gpu_acts = [n for n in act_nodes if n.gpu > 0]
            if gpu_acts:
                g_u = [sum(j["gpu"] * j["gamma"] for j in n.running) / n.gpu for n in gpu_acts]
                gpu_util_sum += sum(g_u) / len(g_u)

    finished = [j for j in jobs if j["end"] is not None and j["end"] <= T_END]
    waits = np.array([j["start"] - j["arrival"] for j in finished]) if finished else np.array([0.0])
    jct = np.array([j["end"] - j["arrival"] for j in finished]) if finished else np.array([0.0])
    slow = np.array([(j["end"] - j["arrival"]) / j["dur"] for j in finished]) if finished else np.array([1.0])

    return dict(
        policy=policy,
        jobs=len(jobs),
        completed=len(finished),
        energy_kwh=energy_wh / 1000.0,
        mean_active_nodes=active_node_sum / steps,
        mean_cpu_util_active=cpu_util_sum / steps,
        mean_gpu_util_active=gpu_util_sum / steps,
        mean_wait=float(waits.mean()),
        p95_wait=float(np.percentile(waits, 95)),
        mean_jct=float(jct.mean()),
        mean_slowdown=float(slow.mean()),
        wakes=wakes,
    )
