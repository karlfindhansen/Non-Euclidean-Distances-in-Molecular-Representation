import gc
import signal
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl

from src.non_euclidean import Grassmann, Riemann, PersistentHomology
from src.optimal_transport import Wasserstein, REMatch

# ── Experimental parameters ──────────────────────────────────────────────────

SEEDS = [42, 123, 456]
NUM_ATOMS = 24
#RUN_TIMEOUT_SECONDS = 600  # per-seed timeout; bail after first timeout in a cell

# D-sweep: varies descriptor dimension at fixed N (reproduces Table 16 with seed averaging)
D_SWEEP_N = 50
D_SWEEP_DIMS = [256, 512, 1024, 2048]

# N-sweep: varies dataset size at fixed D (new table answering RQ on scalability in N)
N_SWEEP_D = 256
N_SWEEP_NS = [50, 100, 200, 500]

# PH already has N-scaling in Figure 29, so skip it in the N-sweep by default
INCLUDE_PH_IN_N_SWEEP = True


# ── Timeout machinery (POSIX / macOS only) ───────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


# ── Dataset generation ───────────────────────────────────────────────────────

def generate_dataset(
    num_molecules: int,
    num_atoms: int,
    dimension: int,
    seed: int,
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = []
    for i in range(num_molecules):
        data.append({
            "molecule_id": f"mol_{i}",
            "num_atoms": num_atoms,
            "atomic_numbers": [6] * num_atoms,
            "coordinates": rng.uniform(-5.0, 5.0, size=(num_atoms, 3)).tolist(),
            "soap_matrix": rng.uniform(0.1, 1.0, size=(num_atoms, dimension)).tolist(),
        })
    return pl.DataFrame(data)


def compute_euclidean_matrix(df: pl.DataFrame, descriptor: str = "soap") -> np.ndarray:
    matrices = df[f"{descriptor}_matrix"].to_list()
    vecs = np.array([np.mean(mat, axis=0) for mat in matrices])
    diffs = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]
    return np.sqrt(np.sum(diffs ** 2, axis=-1))


# ── Profiling ────────────────────────────────────────────────────────────────

def profile_single_run(
    func: Any,
    kwargs: Dict,
) -> Dict:
    gc.collect()

    signal.signal(signal.SIGALRM, _alarm_handler)

    tracemalloc.start()
    t0 = time.perf_counter()
    timed_out = False
    failed = False

    try:
        func(**kwargs)
    except _Timeout:
        timed_out = True
    except Exception as e:
        print(f"  [!] Error: {e}")
        failed = True
    finally:
        signal.alarm(0)
        t1 = time.perf_counter()
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    if timed_out or failed:
        return {"runtime": np.nan, "memory": np.nan, "timed_out": timed_out}

    return {
        "runtime": t1 - t0,
        "memory": peak_mem / (1024 * 1024),
        "timed_out": False,
    }


def _task_templates(include_ph: bool) -> List[Dict]:
    tasks = [
        {
            "name": "Euclidean",
            "func": compute_euclidean_matrix,
            "extra_kwargs": {"descriptor": "soap"},
        },
        {
            "name": "Wasserstein",
            "func": Wasserstein.distance_matrix,
            "extra_kwargs": {"descriptor": "soap", "metric": "sqeuclidean"},
        },
        {
            "name": "REMatch",
            "func": REMatch.distance_matrix,
            "extra_kwargs": {"descriptor": "soap", "alpha": 0.3393},
        },
        {
            "name": "Grassmann",
            "func": Grassmann.distance_matrix,
            "extra_kwargs": {"descriptor": "soap", "distance_type": "geodesic", "k": 3},
        },
        {
            "name": "Log-Euclidean",
            "func": Riemann.distance_matrix,
            "extra_kwargs": {"descriptor": "soap", "distance_type": "log-euclidean", "pca": False},
        },
        {
            "name": "Affine-Invariant",
            "func": Riemann.distance_matrix,
            "extra_kwargs": {"descriptor": "soap", "distance_type": "affine-invariant", "pca": False},
        },
    ]
    if include_ph:
        tasks.append({
            "name": "Persistent Homology",
            "func": PersistentHomology.distance_matrix,
            "extra_kwargs": {
                "descriptor": "coordinates",
                "metric": "sliced_wasserstein",
                "max_homology_dim": 1,
                "homology_dims": (0, 1),
            },
        })
    return tasks


# ── Sweep runner ─────────────────────────────────────────────────────────────

def run_sweep(
    sweep_name: str,
    n_values: List[int],
    d_values: List[int],
    seeds: List[int],
    include_ph: bool = True,
) -> List[Dict]:
    tasks = _task_templates(include_ph)
    records = []

    print(f"\n{'=' * 100}")
    print(f"  {sweep_name}")
    print(f"{'=' * 100}")

    for n in n_values:
        for d in d_values:
            for task in tasks:
                seed_runtimes: List[float] = []
                seed_memories: List[float] = []
                any_timeout = False

                for seed in seeds:
                    print(
                        f"  [{task['name']:<25}]  N={n:>4}  D={d:>4}  seed={seed} ...",
                        end=" ", flush=True,
                    )

                    df = generate_dataset(n, NUM_ATOMS, d, seed)
                    kwargs = {"df": df, **task["extra_kwargs"]}
                    result = profile_single_run(task["func"], kwargs)

                    if result["timed_out"]:
                        any_timeout = True
                        break
                    elif np.isnan(result["runtime"]):
                        print("FAILED")
                        seed_runtimes.append(np.nan)
                        seed_memories.append(np.nan)
                    else:
                        print(f"{result['runtime']:.3f}s  {result['memory']:.1f} MB")
                        seed_runtimes.append(result["runtime"])
                        seed_memories.append(result["memory"])

                records.append({
                    "method": task["name"],
                    "N": n,
                    "D": d,
                    "runtime": np.nan if any_timeout else float(np.nanmean(seed_runtimes)),
                    "memory": np.nan if any_timeout else float(np.nanmean(seed_memories)),
                    "timed_out": any_timeout,
                })

    return records


# ── LaTeX output ─────────────────────────────────────────────────────────────

def _fmt(val: float, timed_out: bool) -> str:
    if timed_out:
        return r">10\,min"
    if np.isnan(val):
        return "---"
    return f"{val:.3f}"


def print_latex_table(
    records: List[Dict],
    vary: str,          # "N" or "D"
    col_values: List[int],
    fixed_key: str,
    fixed_val: int,
    caption: str,
    label: str,
):
    idx = {(r["method"], r["N"], r["D"]): r for r in records}

    methods: List[str] = []
    seen: set = set()
    for r in records:
        if r["method"] not in seen:
            methods.append(r["method"])
            seen.add(r["method"])

    n_cols = len(col_values)
    col_spec = "l|" + "c" * n_cols + "|" + "c" * n_cols
    col_label = vary  # "N" or "D"

    header_cells = " & ".join(
        rf"\textbf{{${col_label}={v}$}}" for v in col_values
    )

    print(f"\n% ─── {label} ─────────────────────────────────────────")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\footnotesize")
    print(rf"\begin{{tabular}}{{{col_spec}}}")
    print(r"\toprule")
    print(
        rf" & \multicolumn{{{n_cols}}}{{c}}{{\textbf{{Runtime Scaling (s)}}}}"
        rf" & \multicolumn{{{n_cols}}}{{c}}{{\textbf{{Peak Memory Consumption (MB)}}}}"
        r" \\"
    )
    print(
        rf"\cmidrule(lr){{2-{n_cols + 1}}}"
        rf" \cmidrule(lr){{{n_cols + 2}-{2 * n_cols + 1}}}"
    )
    print(rf"\textbf{{Method}} & {header_cells} & {header_cells} \\")
    print(r"\midrule")

    for method in methods:
        runtimes, memories = [], []
        for v in col_values:
            key = (method, v, fixed_val) if vary == "N" else (method, fixed_val, v)
            rec = idx.get(key)
            if rec is None:
                runtimes.append("---")
                memories.append("---")
            else:
                runtimes.append(_fmt(rec["runtime"], rec["timed_out"]))
                memories.append(_fmt(rec["memory"], rec["timed_out"]))

        print(
            rf"{method:<26} & {' & '.join(runtimes)}"
            rf" & {' & '.join(memories)} \\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(rf"\caption{{{caption}}}")
    print(rf"\label{{{label}}}")
    print(r"\end{table}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print(f"\nSeeds: {SEEDS}")

    # D-sweep — reproduces Table 16 with 3-seed averaging
    d_records = run_sweep(
        sweep_name=f"D-SWEEP  (N={D_SWEEP_N} fixed, D varies, 3 seeds)",
        n_values=[D_SWEEP_N],
        d_values=D_SWEEP_DIMS,
        seeds=SEEDS,
        include_ph=True,
    )

    # N-sweep — new table answering the RQ on scalability in N
    n_records = run_sweep(
        sweep_name=f"N-SWEEP  (D={N_SWEEP_D} fixed, N varies, 3 seeds)",
        n_values=N_SWEEP_NS,
        d_values=[N_SWEEP_D],
        seeds=SEEDS,
        include_ph=INCLUDE_PH_IN_N_SWEEP,
    )

    all_records = d_records + n_records

    out_dir = Path(__file__).parent
    pl.DataFrame(all_records).write_csv(out_dir / "profiling_results.csv")
    print(f"\nSaved raw results to {out_dir / 'profiling_results.csv'}")

    print("\n\n" + "=" * 100)
    print("LATEX OUTPUT")
    print("=" * 100)

    n_seeds = len(SEEDS)

    print_latex_table(
        records=all_records,
        vary="D",
        col_values=D_SWEEP_DIMS,
        fixed_key="N",
        fixed_val=D_SWEEP_N,
        caption=(
            rf"Computational scaling across descriptor dimensions $D$ at fixed"
            rf" $N={D_SWEEP_N}$ molecules (mean over {n_seeds} seeds). "
            r"For analytical runtimes see \autoref{tab:framework_complexities}."
        ),
        label="tab:computational_scaling",
    )

    print_latex_table(
        records=all_records,
        vary="N",
        col_values=N_SWEEP_NS,
        fixed_key="D",
        fixed_val=N_SWEEP_D,
        caption=(
            rf"Computational scaling across dataset sizes $N$ at fixed"
            rf" $D={N_SWEEP_D}$ descriptor dimension (mean over {n_seeds} seeds). "
            r"Entries marked $>10\,\text{min}$ exceeded the per-run timeout. "
            r"PH is omitted here as its $N$-scaling is shown in \autoref{fig:ph_n_scaling}."
        ),
        label="tab:computational_scaling_N",
    )


if __name__ == "__main__":
    main()
