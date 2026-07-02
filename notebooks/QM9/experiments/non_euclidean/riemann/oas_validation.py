from src.datasets import QM9Dataset
from src.non_euclidean import _feature_matrices_from_df

import numpy as np
import logging
from typing import Any
from sklearn.decomposition import PCA
from sklearn.covariance import oas, ledoit_wolf, empirical_covariance
from sklearn.metrics import pairwise_distances
import polars as pl
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class Riemann:
    
    @classmethod
    def matrix_pca(cls, n_pca, raw_matrices):
        if n_pca is not None:
            logger.info(f"Applying PCA to reduce feature dimension to {n_pca}...")
            
            # Stack all atoms from all molecules into one giant 2D matrix
            stacked_features = np.vstack(raw_matrices)
            pca = PCA(n_components=n_pca)
            stacked_reduced = pca.fit_transform(stacked_features)
            
            # Unstack back into the original list of (N_atoms, n_pca) matrices
            reduced_matrices = []
            current_idx = 0
            for X in raw_matrices:
                n_atoms = X.shape[0]
                reduced_matrices.append(stacked_reduced[current_idx : current_idx + n_atoms, :])
                current_idx += n_atoms

            logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f} (cumulative for {n_pca} components)")
                
            raw_matrices = reduced_matrices

        return raw_matrices

    @classmethod
    def get_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = True,
        estimator: str = 'oas'
    ) -> np.ndarray:
        
        raw_matrices = _feature_matrices_from_df(df, descriptor)

        if pca:
            n_pca = df['num_atoms'].min() - 2
            raw_matrices = cls.matrix_pca(n_pca, raw_matrices)

        spd_matrices = []
        for X in raw_matrices:
            X = np.asarray(X)
            
            # Select covariance estimator
            if estimator == 'oas':
                C, _ = oas(X, assume_centered=False)
            elif estimator == 'ledoit_wolf':
                C, _ = ledoit_wolf(X, assume_centered=False)
            elif estimator == 'empirical':
                C = empirical_covariance(X, assume_centered=False)
            else:
                raise ValueError(f"Unknown estimator: {estimator}")
                
            spd_matrices.append(C)

        for idx, C in enumerate(spd_matrices):
            if not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Matrix at index {idx} failed symmetry validation")
            
            eigvals = np.linalg.eigvalsh(C)
            min_eig = eigvals.min()
            #if min_eig <= 0:
            #    logger.warning(f"Matrix {idx} has eigenvalue <= 0 ({min_eig}) using {estimator}")

        return np.array(spd_matrices)

    @classmethod
    def shrinkage_diagnostics(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = False,
    ) -> pl.DataFrame:
        raw_matrices = _feature_matrices_from_df(df, descriptor)
        if pca:
            n_pca = df['num_atoms'].min() - 2
            raw_matrices = cls.matrix_pca(n_pca, raw_matrices)

        records = []
        for X in raw_matrices:
            X = np.asarray(X, dtype=np.float64)
            n_atoms, D = X.shape
            C, rho = oas(X, assume_centered=False)
            total = np.sum(C ** 2)
            offdiag = total - np.sum(np.diag(C) ** 2)
            records.append({
                "num_atoms": int(n_atoms),
                "D": int(D),
                "oas_rho": float(rho),
                "offdiag_energy_frac": float(offdiag / total) if total > 0 else 0.0,
            })
        return pl.DataFrame(records)

    @classmethod
    def vectorized_spd_matrices(
        cls,
        df: Any,
        descriptor: str = 'soap',
        pca: bool = True,
        estimator: str = 'oas',
        eig_floor: float = 1e-12,
        warn_threshold: float = 1e-6,
    ) -> np.ndarray:
        spd_matrices = cls.get_spd_matrices(df, descriptor, pca, estimator)
        return cls.log_euclidean_vectorize(spd_matrices, eig_floor, warn_threshold)
    
    @classmethod
    def log_euclidean_vectorize(
        cls,
        spd_matrices: np.ndarray,
        eig_floor: float = 1e-12,
        warn_threshold: float = 1e-6,
    ) -> np.ndarray:
        """
        Computes Log-Euclidean vectors from a tensor of SPD matrices.

        Off-diagonal entries are weighted by sqrt(2) so Euclidean dot products
        between the flattened upper triangles preserve the Frobenius inner
        product of the symmetric matrix logarithms.
        """
        spd_matrices = np.asarray(spd_matrices, dtype=np.float64)
        if spd_matrices.ndim != 3 or spd_matrices.shape[1] != spd_matrices.shape[2]:
            raise ValueError(
                "spd_matrices must have shape (n_molecules, d, d) with square matrices."
            )

        _, d, _ = spd_matrices.shape
        triu_idx = np.triu_indices(d)
        weight_matrix = np.where(np.eye(d, dtype=bool), 1.0, np.sqrt(2.0))

        vectorized_dataset = []
        min_eigenvalues = []

        for idx, C in enumerate(spd_matrices):
            if not np.allclose(C, C.T, rtol=1e-5, atol=1e-8):
                raise ValueError(f"Matrix at index {idx} failed symmetry validation.")

            eigenvalues, eigenvectors = np.linalg.eigh(C)
            min_eigenvalues.append(float(eigenvalues.min()))

            eigenvalues = np.clip(eigenvalues, a_min=eig_floor, a_max=None)
            log_C = eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T
            weighted_log_C = log_C * weight_matrix
            vectorized_dataset.append(weighted_log_C[triu_idx])

        global_min_eig = min(min_eigenvalues) if min_eigenvalues else np.nan
        logger.info(
            f"Smallest eigenvalue across SPD dataset: {global_min_eig:.6e}"
        )
        if global_min_eig < warn_threshold:
            logger.warning(
                "Extremely small eigenvalues detected. Verify OAS scaling or centering."
            )
        else:
            logger.info("Minimum eigenvalue looks structurally stable.")

        return np.asarray(vectorized_dataset, dtype=np.float64)

def _logm_sym(C, floor=1e-12):
    w, Q = np.linalg.eigh(C)
    return (Q * np.log(np.clip(w, floor, None))) @ Q.T


def _vech_sqrt2(M):
    d = M.shape[0]
    iu = np.triu_indices(d)
    w = np.where(np.eye(d, dtype=bool), 1.0, np.sqrt(2.0))
    return (M * w)[iu]


def _cv_best_alpha(K_tr, y_tr, alphas, seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best = None
    for a in alphas:
        errs = []
        for itr, iva in kf.split(K_tr):
            m = KernelRidge(alpha=float(a), kernel="precomputed")
            m.fit(K_tr[np.ix_(itr, itr)], y_tr[itr])
            p = m.predict(K_tr[np.ix_(iva, itr)])
            errs.append(np.mean((y_tr[iva] - p) ** 2))
        sc = float(np.mean(errs))
        if best is None or sc < best[1]:
            best = (float(a), sc)
    return best


def run_flaw3_experiment(
    df: pl.DataFrame,
    descriptor: str = 'soap',
    targets: list[str] | None = None,
    seeds: list[int] | None = None,
    alphas: np.ndarray | None = None,
    rbf_mults: tuple = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> pd.DataFrame:
    """
    Ablation ladder to determine where SPD signal lives.

    Representations compared (same OAS covariance C for all):
        mean        — per-molecule mean atom feature (1st moment baseline)
        scalar      — scalar tr(logC)/D
        diagonal    — diag(logC)
        flat_cov    — vech(C)  (second moment, no log map)
        full_tangent— vech(logC) with sqrt(2) off-diagonal weights

    Delta(full_tangent - diagonal) is the decisive number: how much
    cross-channel covariance contributes beyond per-channel log-variances.
    """
    if targets is None:
        targets = [t for t in ["gap", "mu"] if t in df.columns]
    if seeds is None:
        seeds = [42, 123, 456]
    if alphas is None:
        alphas = np.logspace(-4, 4, 17)

    REP_ORDER = ["mean", "scalar", "diagonal", "flat_cov", "full_tangent"]

    # --- shrinkage diagnostic ---
    diag = Riemann.shrinkage_diagnostics(df, descriptor=descriptor, pca=False)
    D = int(diag["D"][0])
    print(f"\n[{descriptor.upper()}] Descriptor dim D={D}, tangent dim={D*(D+1)//2}")
    print(f"  mean OAS rho               = {diag['oas_rho'].mean():.4f}  (1.0 => fully shrunk to scaled identity)")
    print(f"  mean off-diagonal energy    = {diag['offdiag_energy_frac'].mean():.4f}")

    # --- build representations ---
    raw_matrices = _feature_matrices_from_df(df, descriptor)
    mean_feat = np.array([np.asarray(X, dtype=np.float64).mean(axis=0) for X in raw_matrices])

    spd = Riemann.get_spd_matrices(df, descriptor=descriptor, pca=False)

    full, flat, dgl, scl = [], [], [], []
    for C in spd:
        logC = _logm_sym(C)
        full.append(_vech_sqrt2(logC))
        flat.append(_vech_sqrt2(C))
        d = np.diag(logC)
        dgl.append(d)
        scl.append([float(d.mean())])

    reps = {
        "mean":         mean_feat,
        "scalar":       np.asarray(scl, dtype=np.float64),
        "diagonal":     np.asarray(dgl, dtype=np.float64),
        "flat_cov":     np.asarray(flat, dtype=np.float64),
        "full_tangent": np.asarray(full, dtype=np.float64),
    }
    for k in REP_ORDER:
        print(f"  {k:13s} dim = {reps[k].shape[1]}")

    # --- off-diagonal share of between-molecule tangent variance ---
    V = reps["full_tangent"]
    tri = np.triu_indices(D)
    diag_mask = tri[0] == tri[1]
    cv = V.var(axis=0)
    off_share = cv[~diag_mask].sum() / cv.sum()
    print(f"  off-diagonal share of between-molecule tangent variance = {off_share:.4f}")

    # --- evaluate all representations ---
    ys = {t: df[t].to_numpy().astype(np.float64) for t in targets}
    records = []
    for rep_name in REP_ORDER:
        X = reps[rep_name]
        for seed in seeds:
            tr, te = train_test_split(np.arange(X.shape[0]), test_size=0.2, random_state=seed)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd

            Ktr_lin = Xtr @ Xtr.T
            Kte_lin = Xte @ Xtr.T
            Dtr = cdist(Xtr, Xtr)
            Dte = cdist(Xte, Xtr)
            med = np.median(Dtr[np.triu_indices_from(Dtr, k=1)])
            med = med if med > 0 else 1.0

            for target, y in ys.items():
                ytr, yte = y[tr], y[te]

                a, _ = _cv_best_alpha(Ktr_lin, ytr, alphas, seed)
                m = KernelRidge(alpha=a, kernel="precomputed").fit(Ktr_lin, ytr)
                records.append(dict(descriptor=descriptor, rep=rep_name, kind="linear",
                                    target=target, seed=seed, r2=r2_score(yte, m.predict(Kte_lin))))

                best = None
                for mlt in rbf_mults:
                    g = 1.0 / (2.0 * (med * mlt) ** 2)
                    Ktr = np.exp(-g * Dtr ** 2)
                    a, sc = _cv_best_alpha(Ktr, ytr, alphas, seed)
                    if best is None or sc < best[2]:
                        best = (mlt, a, sc)
                mlt, a, _ = best
                g = 1.0 / (2.0 * (med * mlt) ** 2)
                Ktr = np.exp(-g * Dtr ** 2)
                Kte = np.exp(-g * Dte ** 2)
                m = KernelRidge(alpha=a, kernel="precomputed").fit(Ktr, ytr)
                records.append(dict(descriptor=descriptor, rep=rep_name, kind="rbf",
                                    target=target, seed=seed, r2=r2_score(yte, m.predict(Kte))))

    res = pd.DataFrame(records)

    # --- deltas ---
    print(f"\n  [FLAW-3 DELTAS — {descriptor.upper()}]")
    print("  Delta(full_tangent - diagonal) = pure cross-channel covariance contribution")
    for target in ys:
        for kind in ("linear", "rbf"):
            sub = res[(res.target == target) & (res.kind == kind)]
            m = sub.groupby("rep")["r2"].mean()
            print(f"  [{target:12s} | {kind:6s}]  "
                  f"full-diag={m['full_tangent']-m['diagonal']:+.3f}  "
                  f"diag-scalar={m['diagonal']-m['scalar']:+.3f}  "
                  f"flatcov-mean={m['flat_cov']-m['mean']:+.3f}")

    return res


def run_covariance_experiment(df: pl.DataFrame, descriptor: str = 'soap', target_property: str = 'gap') -> pl.DataFrame:
    """
    Evaluates different covariance estimators for SPD manifold generation
    using a Polars DataFrame.
    """
    configs = [
        {"estimator": "empirical", "pca": True},
        {"estimator": "empirical", "pca": False},
        {"estimator": "ledoit_wolf", "pca": True},
        {"estimator": "ledoit_wolf", "pca": False},
        {"estimator": "oas", "pca": True},
        {"estimator": "oas", "pca": False},
    ]

    # Polars syntax to extract the target column as a 1D NumPy array
    y = df.get_column(target_property).to_numpy()
    results = []

    for conf in configs:
        est = conf["estimator"]
        use_pca = conf["pca"]
        print(f"\n--- Running: Descriptor={descriptor.upper()} | Estimator={est.upper()} | PCA={use_pca} ---")

        try:
            # 1. Generate SPD Matrices
            spd_mats = Riemann.get_spd_matrices(
                df=df, descriptor=descriptor, pca=use_pca, estimator=est
            )
            
            # 2. Evaluate Numerical Stability (Condition Number)
            condition_numbers = []
            for C in spd_mats:
                eigvals = np.linalg.eigvalsh(C)
                cond = eigvals.max() / max(eigvals.min(), 1e-15)
                condition_numbers.append(cond)
                
            median_cond = np.median(condition_numbers)
            print(f"Median Condition Number: {median_cond:.2e}")

            # 3. Vectorize via Log-Euclidean mapping
            X_vec = Riemann.log_euclidean_vectorize(spd_mats)
            
            # 4. Property Prediction Task
            X_train, X_test, y_train, y_test = train_test_split(
                X_vec, y, test_size=0.2, random_state=42
            )
            
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            
            print(f"Prediction MAE: {mae:.4f} | R2: {r2:.4f}")
            
            results.append({
                "Descriptor": descriptor,
                "Estimator": est,
                "PCA": use_pca,
                "Median Condition No.": median_cond,
                "MAE": mae,
                "R2": r2,
                "Status": "Success"
            })

        except Exception as e:
            print(f"FAILED: {str(e)}")
            results.append({
                "Descriptor": descriptor,
                "Estimator": est,
                "PCA": use_pca,
                "Median Condition No.": np.nan,
                "MAE": np.nan,
                "R2": np.nan,
                "Status": f"Failed: {type(e).__name__}"
            })

    return pl.DataFrame(results)

if __name__ == "__main__":
    SAMPLE_SIZE = 600
    TARGETS = ["gap", "mu"]

    cov_results = []
    flaw3_results = []

    for descriptor in ["soap", "acsf"]:
        qm9 = QM9Dataset(limit=SAMPLE_SIZE, descriptors=[descriptor])
        df = qm9.load()

        cov_df = run_covariance_experiment(df, descriptor=descriptor, target_property='gap')
        cov_results.append(cov_df)

        f3 = run_flaw3_experiment(df, descriptor=descriptor, targets=TARGETS)
        flaw3_results.append(f3)

    print("\n=== Covariance Estimator Summary ===")
    with pl.Config(tbl_rows=-1, tbl_cols=-1):
        print(pl.concat(cov_results))

    print("\n=== Flaw-3 Ablation Summary ===")
    all_f3 = pd.concat(flaw3_results, ignore_index=True)
    REP_ORDER = ["mean", "scalar", "diagonal", "flat_cov", "full_tangent"]
    agg = (all_f3.groupby(["descriptor", "target", "kind", "rep"])["r2"]
               .mean().reset_index())
    print(agg.pivot_table(index=["descriptor", "target", "kind"], columns="rep", values="r2")
            [REP_ORDER].round(3))