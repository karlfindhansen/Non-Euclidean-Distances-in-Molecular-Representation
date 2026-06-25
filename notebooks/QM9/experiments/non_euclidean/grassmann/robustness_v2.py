"""
Upgraded Robustness Benchmark — Thesis Chapter 4
=================================================
Dual-perturbation stress test across all geometric frameworks evaluated in the thesis.

Dataset   : Balanced C9H16 tripartite (30 strained 3-rings / 30 relaxed 6-rings / 30 acyclic)
Frameworks: Average SOAP | W1 | REMatch | Riemann (AI) | Grassmann k={3,25} | PH-SW | MACE Avg
Modes     : A – Stochastic dihedral rotations (physically grounded conformational noise)
            B – Isotropic bounded jitter with hard-sphere repulsion threshold
Metrics   : MRR (Mean Reciprocal Rank) + Npres (Neighbourhood Preservation Ratio)
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
except NameError:
    # Running via exec() or %run in a notebook – fall back to cwd
    REPO_ROOT = os.path.abspath(
        os.path.join(os.getcwd(), "Anomaly-Detection-in-Molecular-and-Materials-Datasets")
    )
    if not os.path.isdir(os.path.join(REPO_ROOT, "src")):
        REPO_ROOT = os.getcwd()
sys.path.insert(0, REPO_ROOT)

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from ase import Atoms
from dscribe.descriptors import SOAP
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds, rdMolTransforms
from scipy.spatial.distance import pdist, squareform
from typing import Any, Dict, List, Optional, Tuple

from src.non_euclidean import Grassmann, Riemann, PersistentHomology
from src.optimal_transport import Wasserstein, REMatch

RDLogger.DisableLog("rdApp.*")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATASET
# ─────────────────────────────────────────────────────────────────────────────

def _smallest_ring_size(smi: str) -> int:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return -1
    rings = mol.GetRingInfo().AtomRings()
    return min(len(r) for r in rings) if rings else 0


def build_c9h16_tripartite(
    df_full: pl.DataFrame,
    n_per_class: int = 30,
    seed: int = 42,
) -> pl.DataFrame:
    """Returns a balanced 3-class C9H16 slice used throughout the robustness study.

    Classes
    -------
    strained_ring  – smallest ring size == 3 (cyclopropane skeleton, high strain)
    relaxed_ring   – smallest ring size == 6 (cyclohexane skeleton, low strain)
    acyclic        – no rings (open chains)
    """
    rng = np.random.default_rng(seed)

    c9h16 = df_full.filter(pl.col("formula") == "C9H16")
    smiles_list = c9h16["canonical_smiles"].to_list()
    min_ring = [_smallest_ring_size(s) for s in smiles_list]
    c9h16 = c9h16.with_columns(pl.Series("min_ring_size", min_ring, dtype=pl.Int32))

    strained = c9h16.filter(pl.col("min_ring_size") == 3)
    relaxed  = c9h16.filter(pl.col("min_ring_size") == 6)
    acyclic  = c9h16.filter(pl.col("min_ring_size") == 0)

    def _sample(sub: pl.DataFrame, n: int, label: str) -> pl.DataFrame:
        idx = rng.choice(sub.height, size=min(n, sub.height), replace=False)
        return sub[idx.tolist()].with_columns(pl.lit(label).alias("topology_class"))

    df_bal = pl.concat([
        _sample(strained, n_per_class, "strained_ring"),
        _sample(relaxed,  n_per_class, "relaxed_ring"),
        _sample(acyclic,  n_per_class, "acyclic"),
    ], how="diagonal_relaxed").with_row_index("global_idx")

    print(
        f"C9H16 tripartite: "
        f"strained={df_bal.filter(pl.col('topology_class')=='strained_ring').height}, "
        f"relaxed={df_bal.filter(pl.col('topology_class')=='relaxed_ring').height}, "
        f"acyclic={df_bal.filter(pl.col('topology_class')=='acyclic').height}"
    )
    return df_bal


# ─────────────────────────────────────────────────────────────────────────────
# 2.  DESCRIPTOR CALCULATORS
# ─────────────────────────────────────────────────────────────────────────────

def make_soap_calculator(atomic_numbers_lists: List[List[int]]) -> SOAP:
    flat = [z for sub in atomic_numbers_lists for z in sub]
    species = sorted(set(flat))
    return SOAP(species=species, r_cut=5.0, n_max=4, l_max=3, periodic=False, sparse=False)


def compute_soap_matrix(atoms: Atoms, soap: SOAP) -> np.ndarray:
    return soap.create(atoms).astype(np.float64)


def load_mace_calculator(model: str = "medium"):
    from mace.calculators import mace_off
    return mace_off(model=model, device="cpu", default_dtype="float32")


def compute_mace_matrix(atoms: Atoms, mace_calc) -> np.ndarray:
    desc = mace_calc.get_descriptors(atoms)
    if isinstance(desc, (list, tuple)):
        return np.asarray(desc[0], dtype=np.float64)
    return np.asarray(desc, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PERTURBATION MODES
# ─────────────────────────────────────────────────────────────────────────────

def _build_rdkit_mol(atomic_numbers: List[int], coords: np.ndarray) -> Optional[Chem.RWMol]:
    mol = Chem.RWMol()
    for z in atomic_numbers:
        mol.AddAtom(Chem.Atom(int(z)))
    conf = Chem.Conformer(len(atomic_numbers))
    for idx, pos in enumerate(coords):
        conf.SetAtomPosition(idx, (float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        mol.UpdatePropertyCache(strict=False)
        return mol
    except Exception:
        return None


def perturb_dihedral(
    atomic_numbers: List[int],
    coords: np.ndarray,
    sigma_rad: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mode A: Stochastic dihedral rotations along all rotatable bonds.

    sigma_rad  standard deviation of the rotation angle (radians) applied to each
               rotatable bond independently.
    """
    mol = _build_rdkit_mol(atomic_numbers, coords)
    if mol is None:
        return coords.copy()

    conf = mol.GetConformer()
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.rdchem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        b = bond.GetBeginAtomIdx()
        c = bond.GetEndAtomIdx()
        b_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != c]
        c_nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(c).GetNeighbors() if n.GetIdx() != b]
        if not b_nbrs or not c_nbrs:
            continue
        a, d = b_nbrs[0], c_nbrs[0]
        try:
            current = rdMolTransforms.GetDihedralRad(conf, a, b, c, d)
            rdMolTransforms.SetDihedralRad(conf, a, b, c, d, current + rng.normal(0.0, sigma_rad))
        except Exception:
            continue

    return np.array(conf.GetPositions(), dtype=np.float64)


def perturb_bounded_jitter(
    coords: np.ndarray,
    sigma: float,
    min_dist: float = 0.8,
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 8,
) -> np.ndarray:
    """Mode B: Isotropic Gaussian jitter with hard-sphere repulsion threshold.

    Retries up to max_attempts times; falls back to 30 % of sigma if never clean.
    """
    coords = np.asarray(coords, dtype=np.float64)
    for _ in range(max_attempts):
        new = coords + rng.normal(0.0, sigma, coords.shape)
        diffs = new[:, None, :] - new[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)
        if dists.min() >= min_dist:
            return new
    return coords + rng.normal(0.0, sigma * 0.3, coords.shape)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  RECORD BUILDER  (one sigma level)
# ─────────────────────────────────────────────────────────────────────────────

def _build_sigma_records(
    clean_records: List[Dict],
    sigma: float,
    n_perturbations: int,
    mode: str,
    soap: SOAP,
    mace_calc,
    rng: np.random.Generator,
) -> Tuple[List[Dict], float]:
    """Returns (all_records, mean_physical_displacement) for one sigma level."""
    all_records = list(clean_records)
    displacements: List[float] = []

    if sigma == 0.0:
        return all_records, 0.0

    sigma_dihedral = sigma * np.pi  # map σ ∈ [0,0.35] to angle std in radians

    for clean in clean_records:
        clean_coords = np.array(clean["coordinates"], dtype=np.float64)
        for _ in range(n_perturbations):
            if mode == "dihedral":
                new_coords = perturb_dihedral(
                    clean["atomic_numbers"], clean_coords, sigma_dihedral, rng
                )
            else:  # jitter
                new_coords = perturb_bounded_jitter(clean_coords, sigma, rng=rng)

            atom_disp = np.linalg.norm(new_coords - clean_coords, axis=1)
            displacements.extend(atom_disp.tolist())

            pert_atoms = Atoms(numbers=clean["atomic_numbers"], positions=new_coords)
            soap_mat   = compute_soap_matrix(pert_atoms, soap)
            mace_mat   = compute_mace_matrix(pert_atoms, mace_calc) if mace_calc is not None else None

            rec = {
                "parent_idx":     clean["parent_idx"],
                "is_clean":       False,
                "topology_class": clean["topology_class"],
                "num_atoms":      clean["num_atoms"],
                "soap_matrix":    soap_mat.tolist(),
                "average_soap":   soap_mat.mean(axis=0).tolist(),
                "coordinates":    new_coords.tolist(),
                "atomic_numbers": clean["atomic_numbers"],
            }
            if mace_mat is not None:
                rec["mace_matrix"]   = mace_mat.tolist()
                rec["average_mace"]  = mace_mat.mean(axis=0).tolist()
            all_records.append(rec)

    return all_records, float(np.mean(displacements)) if displacements else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DISTANCE MATRICES
# ─────────────────────────────────────────────────────────────────────────────

def _compute_all_matrices(
    df: pl.DataFrame,
    k_grassmann: List[int],
    include_mace: bool,
    rematch_gamma: float = 0.1,
) -> Dict[str, np.ndarray]:
    matrices: Dict[str, np.ndarray] = {}

    X_avg = np.vstack(df["average_soap"].to_list())
    matrices["Average SOAP"] = squareform(pdist(X_avg, metric="euclidean"))

    matrices["Wasserstein W1"] = Wasserstein.distance_matrix(df, "soap", metric="euclidean")
    matrices[f"REMatch (γ={rematch_gamma:.3g})"] = REMatch.distance_matrix(df, "soap", metric="linear", alpha=rematch_gamma)
    matrices["Riemann (AI)"]    = Riemann.distance_matrix(df, "soap", distance_type="affine-invariant", pca=False)

    for k in k_grassmann:
        matrices[f"Grassmann k={k}"] = Grassmann.distance_matrix(df, "soap", distance_type="geodesic", k=k)

    matrices["PH–Sliced W"] = PersistentHomology.distance_matrix(
        df, "soap", metric="sliced-wasserstein", max_homology_dim=1, sw_projections=50
    )

    if include_mace and "mace_matrix" in df.columns:
        X_mace = np.vstack(df["average_mace"].to_list())
        matrices["MACE Average"] = squareform(pdist(X_mace, metric="euclidean"))

    return matrices


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RANK-INVARIANT METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_mrr(
    M: np.ndarray,
    idx_noisy: List[int],
    idx_clean: List[int],
    parent_idx_col: List[int],
) -> float:
    """Mean Reciprocal Rank of the true unperturbed parent."""
    rr_vals = []
    for ni in idx_noisy:
        true_parent = parent_idx_col[ni]      # 0..n_parents-1 index into clean mol list
        dists = M[ni, idx_clean]              # length = n_clean
        sorted_positions = np.argsort(dists)  # position 0 = closest clean mol
        hits = np.where(sorted_positions == true_parent)[0]
        rank = int(hits[0]) + 1 if len(hits) else len(idx_clean)
        rr_vals.append(1.0 / rank)
    return float(np.mean(rr_vals)) if rr_vals else 0.0


def compute_npres(
    M: np.ndarray,
    idx_noisy: List[int],
    topology_col: List[str],
    k: int = 5,
) -> float:
    """Macro-class neighbourhood preservation: fraction of k-NN in same topology class."""
    vals = []
    for ni in idx_noisy:
        dists = M[ni].copy()
        dists[ni] = np.inf
        nn_indices = np.argsort(dists)[:k]
        true_cls = topology_col[ni]
        same = sum(1 for j in nn_indices if topology_col[j] == true_cls)
        vals.append(same / k)
    return float(np.mean(vals)) if vals else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  DISTANCE-MATRIX CACHE
# ─────────────────────────────────────────────────────────────────────────────

def _safe_name(fw: str) -> str:
    """Convert a framework label to a filesystem-safe identifier."""
    return re.sub(r"[^\w\-]", "_", fw).strip("_")


def _config_hash(
    n_per_class: int,
    n_perturbations: int,
    seed: int,
    k_grassmann: List[int],
    sigma_levels: List[float],
    k_npres: int,
    rematch_gamma: float = 0.1,
) -> str:
    payload = json.dumps(
        {
            "n_per_class": n_per_class,
            "n_perturbations": n_perturbations,
            "seed": seed,
            "k_grassmann": sorted(k_grassmann),
            "sigma_levels": sigma_levels,
            "k_npres": k_npres,
            "rematch_gamma": rematch_gamma,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class RobustnessCache:
    """Persists distance matrices and the aggregated results dict to disk.

    Layout
    ------
    cache_dir/
      {config_hash}/
        config.json                              – human-readable run parameters
        D_{mode}_s{sigma_idx:02d}_{fw_safe}.npz – one file per (mode, σ, framework)
        idx_{mode}_s{sigma_idx:02d}.json        – idx_clean / idx_noisy / parent_idx / topology
        results.json                             – final results dict (for plot-only runs)
    """

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def _run_dir(self, config_hash: str) -> str:
        d = os.path.join(self.cache_dir, config_hash)
        os.makedirs(d, exist_ok=True)
        return d

    # ── Config ────────────────────────────────────────────────────────────────

    def save_config(self, config_hash: str, params: Dict[str, Any]) -> None:
        path = os.path.join(self._run_dir(config_hash), "config.json")
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    # ── Per-(mode, sigma) index data ──────────────────────────────────────────

    def _idx_path(self, config_hash: str, mode: str, sigma_idx: int) -> str:
        return os.path.join(
            self._run_dir(config_hash), f"idx_{mode}_s{sigma_idx:02d}.json"
        )

    def save_idx(
        self,
        config_hash: str,
        mode: str,
        sigma_idx: int,
        idx_clean: List[int],
        idx_noisy: List[int],
        parent_idx_col: List[int],
        topology_col: List[str],
    ) -> None:
        with open(self._idx_path(config_hash, mode, sigma_idx), "w") as f:
            json.dump(
                {
                    "idx_clean": idx_clean,
                    "idx_noisy": idx_noisy,
                    "parent_idx_col": parent_idx_col,
                    "topology_col": topology_col,
                },
                f,
            )

    def load_idx(
        self, config_hash: str, mode: str, sigma_idx: int
    ) -> Optional[Dict[str, Any]]:
        p = self._idx_path(config_hash, mode, sigma_idx)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    # ── Distance matrices ─────────────────────────────────────────────────────

    def _matrix_path(
        self, config_hash: str, mode: str, sigma_idx: int, fw: str
    ) -> str:
        return os.path.join(
            self._run_dir(config_hash),
            f"D_{mode}_s{sigma_idx:02d}_{_safe_name(fw)}.npz",
        )

    def save_matrix(
        self,
        config_hash: str,
        mode: str,
        sigma_idx: int,
        fw: str,
        M: np.ndarray,
    ) -> None:
        path = self._matrix_path(config_hash, mode, sigma_idx, fw)
        np.savez_compressed(path, M=M)
        print(f"    cached → {os.path.basename(path)}")

    def load_matrix(
        self, config_hash: str, mode: str, sigma_idx: int, fw: str
    ) -> Optional[np.ndarray]:
        p = self._matrix_path(config_hash, mode, sigma_idx, fw)
        if not os.path.exists(p):
            return None
        return np.load(p)["M"]

    def matrix_exists(
        self, config_hash: str, mode: str, sigma_idx: int, fw: str
    ) -> bool:
        return os.path.exists(self._matrix_path(config_hash, mode, sigma_idx, fw))

    def _results_path(self, config_hash: str) -> str:
        return os.path.join(self._run_dir(config_hash), "results.json")

    def save_results(self, config_hash: str, results: Dict[str, Any]) -> None:
        def _to_python(obj: Any) -> Any:
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_python(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_python(v) for v in obj]
            return obj

        with open(self._results_path(config_hash), "w") as f:
            json.dump(_to_python(results), f, indent=2)
        print(f"Results saved → {self._results_path(config_hash)}")

    def load_results(self, config_hash: str) -> Optional[Dict[str, Any]]:
        p = self._results_path(config_hash)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return json.load(f)

    def list_runs(self) -> List[str]:
        """Return config hashes for all cached runs."""
        if not os.path.isdir(self.cache_dir):
            return []
        return [
            d for d in os.listdir(self.cache_dir)
            if os.path.isdir(os.path.join(self.cache_dir, d))
        ]


def load_saved_results(
    cache_dir: str,
    config_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a previously saved results dict without re-running the benchmark.

    Parameters
    ----------
    cache_dir    : directory passed to RobustnessCache (same as used when running)
    config_hash  : 12-char hash printed during a run; if None, loads the most
                   recently modified run in cache_dir.

    Usage (notebook)
    ----------------
    results = load_saved_results("outputs/robustness_v2/cache")
    plot_metric_vs_sigma(results)
    plot_auc_heatmap(results)
    make_auc_table(results)
    """
    cache = RobustnessCache(cache_dir)
    runs  = cache.list_runs()
    if not runs:
        raise FileNotFoundError(f"No cached runs found in {cache_dir!r}.")

    if config_hash is None:
        # Pick the most recently written results.json
        def _mtime(h: str) -> float:
            p = cache._results_path(h)
            return os.path.getmtime(p) if os.path.exists(p) else 0.0

        config_hash = max(runs, key=_mtime)
        print(f"Loading most recent run: config_hash={config_hash!r}")

    results = cache.load_results(config_hash)
    if results is None:
        raise FileNotFoundError(
            f"results.json not found for config_hash={config_hash!r} in {cache_dir!r}.\n"
            "Run the benchmark first (run_upgraded_robustness with a RobustnessCache)."
        )
    print(f"Loaded results: modes={results['modes']}, frameworks={results['frameworks']}")
    return results


def _optimize_rematch_gamma(
    clean_records: List[Dict],
    soap,
    rng: np.random.Generator,
    gamma_candidates: Optional[List[float]] = None,
    opt_sigmas: Optional[List[float]] = None,
    n_perturbations: int = 5,
    k_npres: int = 5,
) -> float:
    """Grid-search γ (alpha) for REMatch over [1e-3, 1e2]; return best by combined MRR+Npres."""
    if gamma_candidates is None:
        gamma_candidates = np.logspace(-3, 2, 10).tolist()
    if opt_sigmas is None:
        opt_sigmas = [0.05, 0.10, 0.20, 0.30]

    print(f"\nOptimizing REMatch γ over {len(gamma_candidates)} candidates …")
    best_gamma, best_score = gamma_candidates[0], -1.0

    for gamma in gamma_candidates:
        scores = []
        for sigma in opt_sigmas:
            all_rec, _ = _build_sigma_records(
                clean_records, sigma, n_perturbations, "jitter", soap, None, rng
            )
            df_s = pl.DataFrame(all_rec)
            idx_clean = (
                df_s.with_row_index().filter(pl.col("is_clean"))
                .select("index").to_series().to_list()
            )
            idx_noisy = (
                df_s.with_row_index().filter(~pl.col("is_clean"))
                .select("index").to_series().to_list()
            )
            parent_idx_col = df_s["parent_idx"].to_list()
            topology_col   = df_s["topology_class"].to_list()
            
            M     = REMatch.distance_matrix(df_s, "soap", metric="linear", alpha=gamma)
            mrr   = compute_mrr(M, idx_noisy, idx_clean, parent_idx_col)
            npres = compute_npres(M, idx_noisy, topology_col, k=k_npres)
            scores.append((mrr + npres) / 2.0)

        score = float(np.mean(scores))
        print(f"    γ = {gamma:.4g}  →  score = {score:.4f}")
        if score > best_score:
            best_score, best_gamma = score, gamma

    print(f"  → Best γ = {best_gamma:.4g}  (score = {best_score:.4f})")
    return best_gamma


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────

def run_upgraded_robustness(
    df_full: pl.DataFrame,
    n_per_class: int = 30,
    n_perturbations: int = 15,
    k_grassmann: List[int] = [3, 25],
    k_npres: int = 5,
    include_mace: bool = True,
    modes: List[str] = ["dihedral", "jitter"],
    seed: int = 42,
    sigma_levels: Optional[List[float]] = None,
    cache: Optional[RobustnessCache] = None,
) -> Dict[str, Any]:
    if sigma_levels is None:
        sigma_levels = [0.00, 0.01, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    if cache is None:
        cache = RobustnessCache(_DEFAULT_CACHE_DIR)
        print(f"Cache auto-created at {_DEFAULT_CACHE_DIR!r}")

    rng = np.random.default_rng(seed)

    # ── Dataset ──────────────────────────────────────────────────────────────
    df_parents = build_c9h16_tripartite(df_full, n_per_class=n_per_class, seed=seed)
    soap = make_soap_calculator(df_parents["atomic_numbers"].to_list())

    mace_calc = None
    if include_mace:
        try:
            mace_calc = load_mace_calculator()
            print("MACE calculator loaded.")
        except Exception as e:
            print(f"MACE unavailable ({e}). Skipping MACE framework.")

    # ── Precompute clean reference records ────────────────────────────────────
    clean_records: List[Dict] = []
    for i, row in enumerate(df_parents.iter_rows(named=True)):
        nums   = row["atomic_numbers"]
        coords = np.array(row["coordinates"], dtype=np.float64)
        atoms  = Atoms(numbers=nums, positions=coords)
        soap_mat = compute_soap_matrix(atoms, soap)
        rec = {
            "parent_idx":     i,
            "is_clean":       True,
            "topology_class": row["topology_class"],
            "num_atoms":      len(nums),
            "soap_matrix":    soap_mat.tolist(),
            "average_soap":   soap_mat.mean(axis=0).tolist(),
            "coordinates":    coords.tolist(),
            "atomic_numbers": nums,
        }
        if mace_calc is not None:
            mace_mat = compute_mace_matrix(atoms, mace_calc)
            rec["mace_matrix"]  = mace_mat.tolist()
            rec["average_mace"] = mace_mat.mean(axis=0).tolist()
        clean_records.append(rec)

    # ── Optimise REMatch γ (separate rng so main benchmark is unaffected) ─────
    opt_rng = np.random.default_rng(seed ^ 0xCAFEBABE)
    rematch_gamma = _optimize_rematch_gamma(clean_records, soap, opt_rng, k_npres=k_npres)
    rematch_label = f"REMatch (γ={rematch_gamma:.3g})"

    # ── Config hash & cache ───────────────────────────────────────────────────
    cfg_hash = _config_hash(n_per_class, n_perturbations, seed, k_grassmann, sigma_levels, k_npres, rematch_gamma)
    if cache is not None:
        cache.save_config(cfg_hash, {
            "n_per_class": n_per_class, "n_perturbations": n_perturbations,
            "seed": seed, "k_grassmann": k_grassmann,
            "sigma_levels": sigma_levels, "k_npres": k_npres,
            "modes": modes, "include_mace": include_mace,
            "rematch_gamma": rematch_gamma,
        })
        print(f"Cache active — config_hash={cfg_hash!r}  ({cache.cache_dir})")

    # ── Framework registry ────────────────────────────────────────────────────
    frameworks = (
        ["Average SOAP", "Wasserstein W1", rematch_label, "Riemann (AI)"]
        + [f"Grassmann k={k}" for k in k_grassmann]
        + ["PH–Sliced W"]
        + (["MACE Average"] if (mace_calc is not None) else [])
    )

    # ── Results storage ───────────────────────────────────────────────────────
    results: Dict[str, Any] = {
        "sigma_levels":  sigma_levels,
        "frameworks":    frameworks,
        "k_npres":       k_npres,
        "modes":         modes,
        "rematch_gamma": rematch_gamma,
        "metrics":       {},
    }

    for mode in modes:
        results["metrics"][mode] = {
            fw: {"mrr": [], "npres": []} for fw in frameworks
        }

    # ── Main loop ─────────────────────────────────────────────────────────────
    # ── Main loop ─────────────────────────────────────────────────────────────
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Perturbation mode: {'A – Dihedral rotations' if mode=='dihedral' else 'B – Bounded jitter'}")
        print(f"{'='*60}")

        for s_idx, sigma in enumerate(sigma_levels):
            print(f"  σ = {sigma:.3f} …", end=" ", flush=True)

            # ── Check whether all matrices for this (mode, sigma) are cached ──
            all_cached = cache is not None and all(
                cache.matrix_exists(cfg_hash, mode, s_idx, fw) for fw in frameworks
            )

            if all_cached:
                idx_data       = cache.load_idx(cfg_hash, mode, s_idx)
                idx_clean      = idx_data["idx_clean"]
                idx_noisy      = idx_data["idx_noisy"]
                parent_idx_col = idx_data["parent_idx_col"]
                topology_col   = idx_data["topology_col"]

                for fw in frameworks:
                    M = cache.load_matrix(cfg_hash, mode, s_idx, fw)
                    if sigma == 0.0:
                        mrr, npres = 1.0, 1.0
                    else:
                        mrr   = compute_mrr(M, idx_noisy, idx_clean, parent_idx_col)
                        npres = compute_npres(M, idx_noisy, topology_col, k=k_npres)
                    results["metrics"][mode][fw]["mrr"].append(mrr)
                    results["metrics"][mode][fw]["npres"].append(npres)
                print("(from cache)")
                continue

            # ── Compute fresh ─────────────────────────────────────────────────
            if sigma == 0.0:
                # Baseline uses unperturbed clean records; no noisy pairs exist
                df_sigma = pl.DataFrame(clean_records)
                idx_clean = list(range(len(clean_records)))
                idx_noisy = []
                parent_idx_col = df_sigma["parent_idx"].to_list()
                topology_col   = df_sigma["topology_class"].to_list()
                mean_disp = 0.0
            else:
                all_records, mean_disp = _build_sigma_records(
                    clean_records, sigma, n_perturbations, mode,
                    soap, mace_calc, rng
                )
                df_sigma = pl.DataFrame(all_records)
                idx_clean = (
                    df_sigma.with_row_index()
                    .filter(pl.col("is_clean"))
                    .select("index").to_series().to_list()
                )
                idx_noisy = (
                    df_sigma.with_row_index()
                    .filter(~pl.col("is_clean"))
                    .select("index").to_series().to_list()
                )
                parent_idx_col = df_sigma["parent_idx"].to_list()
                topology_col   = df_sigma["topology_class"].to_list()

            if cache is not None:
                cache.save_idx(cfg_hash, mode, s_idx,
                               idx_clean, idx_noisy, parent_idx_col, topology_col)

            matrices = _compute_all_matrices(
                df_sigma, k_grassmann, include_mace and mace_calc is not None, rematch_gamma
            )

            for fw in frameworks:
                if fw in matrices:
                    M = matrices[fw]
                    if cache is not None:
                        cache.save_matrix(cfg_hash, mode, s_idx, fw, M)
                    
                    if sigma == 0.0:
                        mrr, npres = 1.0, 1.0
                    else:
                        mrr   = compute_mrr(M, idx_noisy, idx_clean, parent_idx_col)
                        npres = compute_npres(M, idx_noisy, topology_col, k=k_npres)
                else:
                    mrr, npres = np.nan, np.nan
                results["metrics"][mode][fw]["mrr"].append(mrr)
                results["metrics"][mode][fw]["npres"].append(npres)

            if sigma == 0.0:
                print("(baseline matrix saved)")
            else:
                print(f"displacement={mean_disp:.3f} Å")

    if cache is not None:
        cache.save_results(cfg_hash, results)

    print("\nBenchmark complete.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 8.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = {
    "Average SOAP":   "#64748b",
    "Wasserstein W1": "#b45309",
    "REMatch":        "#6d28d9",
    "Riemann (AI)":   "#dc2626",
    "Grassmann k=3":  "#1d4ed8",
    "Grassmann k=25": "#0ea5e9",
    "PH–Sliced W":    "#059669",
    "MACE Average":   "#d97706",
}

_LINESTYLE = {
    "Average SOAP":   "--",
    "Wasserstein W1": "--",
    "REMatch":        "--",
    "Riemann (AI)":   "-",
    "Grassmann k=3":  "-",
    "Grassmann k=25": "-",
    "PH–Sliced W":    "-",
    "MACE Average":   "-.",
}

_LINEWIDTH = {
    "Riemann (AI)":    2.5,
    "Grassmann k=25":  2.5,
}

_MODE_LABELS = {
    "dihedral": "Mode A — Dihedral Rotations",
    "jitter":   "Mode B — Isotropic Bounded Jitter",
}

_METRIC_LABELS = {
    "mrr":   "Mean Reciprocal Rank (MRR)",
    "npres": "Neighbourhood Preservation Ratio ($N_{\\mathrm{pres}}$)",
}


def _fw_style(fw: str, mapping: dict, default: Any) -> Any:
    """Exact-key lookup, falling back to prefix match for dynamic labels like 'REMatch (γ=...)'."""
    if fw in mapping:
        return mapping[fw]
    for k, v in mapping.items():
        if fw.startswith(k):
            return v
    return default


def plot_metric_vs_sigma(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """2 rows (MRR, Npres) × 2 cols (Mode A, Mode B) publication figure."""
    sigma_levels = results["sigma_levels"]
    frameworks   = results["frameworks"]
    modes        = results["modes"]
    metrics_data = results["metrics"]

    metric_keys = ["mrr", "npres"]
    n_rows, n_cols = len(metric_keys), len(modes)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        dpi=300,
        sharex=True,
    )
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]

    for row_idx, metric in enumerate(metric_keys):
        for col_idx, mode in enumerate(modes):
            ax = axes[row_idx][col_idx]

            for fw in frameworks:
                vals = metrics_data[mode][fw][metric]
                color = _fw_style(fw, _PALETTE, "#000000")
                ls    = _fw_style(fw, _LINESTYLE, "-")
                lw    = _fw_style(fw, _LINEWIDTH, 1.6)

                ax.plot(
                    sigma_levels, vals,
                    label=fw, color=color,
                    linestyle=ls, linewidth=lw,
                    marker="o", markersize=3.5,
                    zorder=5 if lw > 2 else 2,
                )

            if row_idx == 0:
                ax.set_title(_MODE_LABELS.get(mode, mode), fontsize=11, fontweight="bold", pad=8)
            if col_idx == 0:
                ax.set_ylabel(_METRIC_LABELS[metric], fontsize=10)
            if row_idx == n_rows - 1:
                ax.set_xlabel("Perturbation Magnitude (σ)", fontsize=10)

            ax.set_xlim([sigma_levels[0], sigma_levels[-1]])
            ax.set_ylim([0.0, 1.05])
            ax.grid(True, linestyle=":", alpha=0.55)

            if row_idx == 0 and col_idx == n_cols - 1:
                ax.legend(
                    frameon=True, facecolor="white", edgecolor="#e2e8f0",
                    loc="lower left", fontsize=8, ncol=1,
                )

    fig.suptitle(
        "Representation Robustness: Rank-Invariant Metrics under Structural Perturbation",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved figure → {save_path}")
    plt.show()


def _auc(vals: List[float], sigma_levels: List[float]) -> float:
    """Trapezoidal AUC normalised by sigma range, ignoring NaN."""
    x = np.array(sigma_levels, dtype=float)
    y = np.array(vals, dtype=float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    return float(np.trapezoid(y[mask], x[mask]) / (x[mask][-1] - x[mask][0]))


def make_auc_table(results: Dict[str, Any], print_latex: bool = True) -> pl.DataFrame:
    """Builds a summary table of AUC-MRR and AUC-Npres per framework × mode."""
    sigma_levels = results["sigma_levels"]
    frameworks   = results["frameworks"]
    modes        = results["modes"]
    metrics_data = results["metrics"]

    rows = []
    for fw in frameworks:
        row: Dict[str, Any] = {"Framework": fw}
        for mode in modes:
            for metric in ["mrr", "npres"]:
                vals = metrics_data[mode][fw][metric]
                key  = f"{mode.capitalize()[:1]} {metric.upper()}"
                row[key] = round(_auc(vals, sigma_levels), 4)
        rows.append(row)

    df_table = pl.DataFrame(rows).sort("D MRR", descending=True)
    print("\nAUC Summary Table")
    print("─" * 60)
    print(df_table)

    if print_latex:
        print("\nLaTeX snippet:")
        print("\\begin{tabular}{l" + "r" * (len(df_table.columns) - 1) + "}")
        print("\\toprule")
        print(" & ".join(df_table.columns) + " \\\\")
        print("\\midrule")
        for row in df_table.iter_rows(named=True):
            cells = [str(row["Framework"])] + [f"{v:.4f}" for k, v in row.items() if k != "Framework"]
            print(" & ".join(cells) + " \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    return df_table


_MODE_SHORT = {"dihedral": "Dihedral", "jitter": "Jitter"}
def plot_auc_heatmap(results: Dict[str, Any], save_path: Optional[str] = None) -> None:
    """Heatmap of normalised AUC values (frameworks × metric-mode combinations)."""
    sigma_levels = results["sigma_levels"]
    frameworks   = results["frameworks"]
    modes        = results["modes"]
    metrics_data = results["metrics"]

    col_labels = [f"{_MODE_SHORT.get(m, m)}\n{met.upper()}" for m in modes for met in ["mrr", "npres"]]
    data = np.zeros((len(frameworks), len(col_labels)))

    col_idx = 0
    for mode in modes:
        for metric in ["mrr", "npres"]:
            for fw_idx, fw in enumerate(frameworks):
                vals = metrics_data[mode][fw][metric]
                data[fw_idx, col_idx] = _auc(vals, sigma_levels)
            col_idx += 1

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 1.4), max(4, len(frameworks) * 0.55)), dpi=300)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(data, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")

    # Set major ticks for labels
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8)
    ax.set_yticks(range(len(frameworks)))
    ax.set_yticklabels(frameworks, fontsize=9)

    # --- FIX: Align grid lines to cell borders instead of centers ---
    ax.grid(False)  # Disable the major grid lines striking through text centers
    
    # Define minor ticks at the half-integer boundaries between pixels
    ax.set_xticks(np.arange(len(col_labels)) + 0.5, minor=True)
    ax.set_yticks(np.arange(len(frameworks)) + 0.5, minor=True)
    
    # Draw the grid exclusively on the minor boundaries
    ax.grid(True, which="minor", color="white", linestyle="-", linewidth=1.5)
    # -----------------------------------------------------------------

    for i in range(len(frameworks)):
        for j in range(len(col_labels)):
            v = data[i, j]
            text_color = "white" if v < 0.4 or v > 0.85 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color=text_color)

    plt.colorbar(im, ax=ax, label="Normalised AUC")
    ax.set_title("Robustness AUC Heatmap (higher = more stable)", fontweight="bold", pad=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved heatmap → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 10.  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
#
# Full run (compute + cache + plot):
#   _run_default(df)
#
# Plots only (no recomputation):
#   results = load_saved_results("outputs/robustness_v2/cache")
#   plot_metric_vs_sigma(results)
#   plot_auc_heatmap(results)
#   make_auc_table(results)
#
# Expects `df` to be the full QM9 Polars DataFrame already in scope.
# Set INCLUDE_MACE = False to skip MACE (much faster).
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_OUT_DIR   = os.path.join(REPO_ROOT, "outputs", "robustness_v2")
_DEFAULT_CACHE_DIR = os.path.join(_DEFAULT_OUT_DIR, "cache")


def _run_default(df_full: pl.DataFrame) -> Dict[str, Any]:
    """Convenience wrapper: call this from a notebook after loading df.

    Computes the full benchmark (resuming from cache if available) then
    generates all plots and the LaTeX table.  Returns the results dict so
    you can do further analysis in the notebook.
    """
    INCLUDE_MACE = False   # flip to True once SOAP results look good
    os.makedirs(_DEFAULT_OUT_DIR, exist_ok=True)

    cache = RobustnessCache(_DEFAULT_CACHE_DIR)

    final_results = run_upgraded_robustness(
        df_full         = df_full,
        n_per_class     = 30,
        n_perturbations = 15,
        k_grassmann     = [3, 25],
        include_mace    = INCLUDE_MACE,
        modes           = ["dihedral", "jitter"],
        seed            = 42,
        cache           = cache,
    )
    _render_outputs(final_results)
    return final_results


def _render_outputs(results: Dict[str, Any]) -> None:
    """Generate all figures and the LaTeX table from a results dict."""
    os.makedirs(_DEFAULT_OUT_DIR, exist_ok=True)
    plot_metric_vs_sigma(
        results,
        save_path=os.path.join(_DEFAULT_OUT_DIR, "robustness_mrr_npres.png"),
    )
    plot_auc_heatmap(
        results,
        save_path=os.path.join(_DEFAULT_OUT_DIR, "robustness_auc_heatmap.png"),
    )
    make_auc_table(results, print_latex=True)


if __name__ == "__main__":
    import polars as _pl
    _df = _pl.read_parquet(os.path.join(REPO_ROOT, "data", "QM9", "dataset_cleaned.parquet"))
    _run_default(_df)
