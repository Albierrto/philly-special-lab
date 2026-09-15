"""The fitted plate-appearance model (hrboard/model.json) as plain numpy, so the daily build needs no fitting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load(path: Path | None = None) -> dict:
    return json.loads((path or HERE / "model.json").read_text())


ORDER = ("out", "k", "bb", "s1", "xb", "hr")


def predict(X: pd.DataFrame, m: dict, O: np.ndarray | None = None, order=ORDER) -> np.ndarray:
    """Class probabilities in `order`. O: per-row log league rates in m['classes'] order (dataset.offsets)."""
    Z = X[m["columns"]].to_numpy(dtype="float64") @ np.array(m["coef"]).T + np.array(m["intercept"])
    if O is not None: Z = Z + O
    Z -= Z.max(axis=1, keepdims=True)
    P = np.exp(Z); P /= P.sum(axis=1, keepdims=True)
    idx = [m["classes"].index(c) for c in order]
    return P[:, idx]


def fit_mnl(X: np.ndarray, y: np.ndarray, O: np.ndarray, l2: float = 1.0, classes=ORDER) -> dict:
    """Multinomial logit with a per-row offset, by L-BFGS. y: class labels. The first class is the reference only in
    the sense that all six rows of coefficients are fitted with a ridge penalty (sklearn's convention)."""
    from scipy.optimize import minimize
    n, p = X.shape; K = len(classes)
    Y = np.zeros((n, K)); Y[np.arange(n), [classes.index(c) for c in y]] = 1
    def f(w):
        W = w[:K * p].reshape(K, p); b = w[K * p:]
        Z = X @ W.T + b + O
        Z -= Z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(Z).sum(axis=1, keepdims=True))
        logP = Z - lse
        loss = -(Y * logP).sum() + 0.5 * l2 * (W ** 2).sum()
        G = np.exp(logP) - Y
        gW = G.T @ X + l2 * W; gb = G.sum(axis=0)
        return loss / n, np.r_[gW.ravel(), gb] / n
    w0 = np.zeros(K * p + K)
    r = minimize(f, w0, jac=True, method="L-BFGS-B", options=dict(maxiter=3000, gtol=1e-7))
    W = r.x[:K * p].reshape(K, p); b = r.x[K * p:]
    return dict(classes=list(classes), coef=W.tolist(), intercept=b.tolist(), converged=bool(r.success), nll=float(r.fun))


def league(m: dict) -> pd.DataFrame:
    return pd.DataFrame(m["lg"]).set_index("season")
