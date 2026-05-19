import typing as _t

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def _normalize_objective(objective: str) -> str:
    obj = str(objective).lower().strip()
    if obj not in {"ml", "reml"}:
        obj = "ml"
    return obj


def _compute_knn_weights(
    distances: np.ndarray,
    weighting: str,
    eps: float,
    gaussian_sigma: _t.Optional[float] = None,
) -> np.ndarray:
    mode = str(weighting).lower().strip()
    if mode == "inverse_distance":
        return 1.0 / (distances + eps)
    if mode == "exponential":
        return np.exp(-distances)
    if mode == "gaussian":
        if gaussian_sigma is None:
            sigma = np.maximum(np.mean(distances, axis=1, keepdims=True), eps)
        else:
            sigma = max(float(gaussian_sigma), eps)
        return np.exp(-(distances**2) / (2.0 * (sigma**2)))
    return 1.0 / (distances + eps)


def compute_group_blup_random_effects(
    residuals: np.ndarray,
    groups: _t.Sequence[_t.Any],
    sigma_b2: float,
    sigma_e2: float,
) -> tuple[dict[str, float], np.ndarray]:
    residuals_arr = np.asarray(residuals, dtype=float).reshape(-1)
    groups_arr = np.asarray(groups).astype(str)

    if residuals_arr.shape[0] != groups_arr.shape[0]:
        n = min(residuals_arr.shape[0], groups_arr.shape[0])
        residuals_arr = residuals_arr[:n]
        groups_arr = groups_arr[:n]

    group_effects: dict[str, float] = {}
    random_effects = np.zeros_like(residuals_arr)

    df = pd.DataFrame({"group": groups_arr, "residual": residuals_arr})
    agg = df.groupby("group")["residual"].agg(["sum", "count"])

    for group, row in agg.iterrows():
        n_g = float(row["count"])
        sum_r = float(row["sum"])
        denom = sigma_e2 + n_g * sigma_b2
        if denom <= 0.0:
            effect = 0.0
        else:
            effect = (sigma_b2 / denom) * sum_r
        group_effects[str(group)] = float(effect)

    for idx, g in enumerate(groups_arr):
        random_effects[idx] = group_effects.get(str(g), 0.0)

    return group_effects, random_effects


def compute_knn_blup_random_effects(
    residuals_train: np.ndarray,
    neighbor_features_train: np.ndarray,
    neighbor_features_query: np.ndarray,
    k: int,
    metric: str,
    eps: float,
    weighting: str,
    sigma_b2: float,
    sigma_e2: float,
    gaussian_sigma: _t.Optional[float] = None,
) -> tuple[np.ndarray, NearestNeighbors, int]:
    residuals = np.asarray(residuals_train, dtype=float).reshape(-1)
    train_features = np.asarray(neighbor_features_train, dtype=float)
    query_features = np.asarray(neighbor_features_query, dtype=float)

    if residuals.shape[0] != train_features.shape[0]:
        n = min(residuals.shape[0], train_features.shape[0])
        residuals = residuals[:n]
        train_features = train_features[:n]

    n_neighbors = int(max(1, min(int(k), train_features.shape[0])))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric=str(metric))
    nn.fit(train_features)

    distances, indices = nn.kneighbors(query_features, return_distance=True)
    weights = _compute_knn_weights(
        distances=distances,
        weighting=weighting,
        eps=float(eps),
        gaussian_sigma=gaussian_sigma,
    )

    numer = np.sum(weights * residuals[indices], axis=1)
    denom = np.maximum(np.sum(weights, axis=1), float(eps))
    weighted_mean = numer / denom

    n_eff = (denom**2) / np.maximum(np.sum(weights**2, axis=1), float(eps))
    shrink = (n_eff * sigma_b2) / np.maximum(sigma_e2 + n_eff * sigma_b2, float(eps))
    random_effects = shrink * weighted_mean

    return random_effects.astype(float), nn, n_neighbors


def compute_group_log_likelihood(
    residuals: np.ndarray,
    groups: _t.Sequence[_t.Any],
    sigma_b2: float,
    sigma_e2: float,
    objective: str = "ml",
    n_fixed_effects: int = 0,
) -> float:
    obj = _normalize_objective(objective)
    residuals_arr = np.asarray(residuals, dtype=float).reshape(-1)
    groups_arr = np.asarray(groups).astype(str)

    if residuals_arr.shape[0] == 0:
        return float("-inf")

    ll = 0.0
    df = pd.DataFrame({"group": groups_arr, "residual": residuals_arr})

    for _, group_df in df.groupby("group"):
        r = group_df["residual"].to_numpy(dtype=float)
        n_g = r.shape[0]

        a = float(sigma_e2)
        b = float(sigma_b2)
        if a <= 0.0 or b < 0.0:
            return float("-inf")

        logdet = (n_g - 1) * np.log(a) + np.log(a + n_g * b)

        sum_r = np.sum(r)
        quad = (np.sum(r**2) / a) - ((b / (a * (a + n_g * b))) * (sum_r**2))

        ll += -0.5 * (n_g * np.log(2.0 * np.pi) + logdet + quad)

    if obj == "reml":
        n = residuals_arr.shape[0]
        p = int(max(0, n_fixed_effects))
        if n - p > 0:
            ll -= 0.5 * p * np.log(2.0 * np.pi)

    return float(ll)


def compute_knn_pseudo_log_likelihood(
    residuals: np.ndarray,
    sigma_b2: float,
    sigma_e2: float,
    objective: str = "ml",
    n_fixed_effects: int = 0,
) -> float:
    obj = _normalize_objective(objective)
    r = np.asarray(residuals, dtype=float).reshape(-1)
    n = r.shape[0]
    if n == 0:
        return float("-inf")

    v = float(sigma_b2 + sigma_e2)
    if v <= 0.0:
        return float("-inf")

    ll = -0.5 * n * np.log(2.0 * np.pi * v) - 0.5 * np.sum((r**2) / v)
    if obj == "reml":
        p = int(max(0, n_fixed_effects))
        if n - p > 0:
            ll -= 0.5 * p * np.log(v)
    return float(ll)


def estimate_variance_components_iterative(
    residuals: np.ndarray,
    random_effects: np.ndarray,
    objective: str = "ml",
    n_fixed_effects: int = 0,
    sigma_b2_init: float = 1.0,
    sigma_e2_init: float = 1.0,
    max_iter: int = 25,
    tol: float = 1e-6,
    var_floor: float = 1e-8,
    groups: _t.Optional[_t.Sequence[_t.Any]] = None,
    is_knn: bool = False,
) -> tuple[float, float, list[dict[str, float]]]:
    _ = _normalize_objective(objective)

    r = np.asarray(residuals, dtype=float).reshape(-1)
    b = np.asarray(random_effects, dtype=float).reshape(-1)

    sigma_b2 = max(float(sigma_b2_init), float(var_floor))
    sigma_e2 = max(float(sigma_e2_init), float(var_floor))

    history: list[dict[str, float]] = []

    n = max(1, int(r.shape[0]))
    p = int(max(0, n_fixed_effects))
    denom_ml = float(n)
    denom_reml = float(max(1, n - p))

    for iter_idx in range(1, int(max_iter) + 1):
        prev_b2 = sigma_b2
        prev_e2 = sigma_e2

        if is_knn:
            sigma_b2 = max(float(np.mean(b**2)), float(var_floor))
            denom = denom_reml if str(objective).lower() == "reml" else denom_ml
            sigma_e2 = max(float(np.sum((r - b) ** 2) / denom), float(var_floor))
            ll = compute_knn_pseudo_log_likelihood(
                residuals=r,
                sigma_b2=sigma_b2,
                sigma_e2=sigma_e2,
                objective=objective,
                n_fixed_effects=n_fixed_effects,
            )
        else:
            sigma_b2 = max(float(np.mean(b**2)), float(var_floor))
            denom = denom_reml if str(objective).lower() == "reml" else denom_ml
            sigma_e2 = max(float(np.sum((r - b) ** 2) / denom), float(var_floor))
            ll = compute_group_log_likelihood(
                residuals=r,
                groups=groups if groups is not None else np.arange(n),
                sigma_b2=sigma_b2,
                sigma_e2=sigma_e2,
                objective=objective,
                n_fixed_effects=n_fixed_effects,
            )

        history.append(
            {
                "iter": float(iter_idx),
                "sigma_b2": float(sigma_b2),
                "sigma_e2": float(sigma_e2),
                "log_likelihood": float(ll),
            }
        )

        delta = max(abs(sigma_b2 - prev_b2), abs(sigma_e2 - prev_e2))
        if delta <= float(tol):
            break

    return float(sigma_b2), float(sigma_e2), history


class BLUPMixedEffectsRegressor:
    def __init__(
        self,
        base_model,
        mode: str,
        group_by: str,
        group_effects: _t.Optional[dict[str, float]] = None,
        sigma_b2: float = 1.0,
        sigma_e2: float = 1.0,
        knn_model: _t.Optional[NearestNeighbors] = None,
        knn_mode: _t.Optional[str] = None,
        knn_metric: str = "euclidean",
        knn_eps: float = 1e-8,
        knn_weighting: str = "inverse_distance",
        knn_gaussian_sigma: _t.Optional[float] = None,
        knn_residuals_train: _t.Optional[np.ndarray] = None,
        knn_neighbor_features_train: _t.Optional[np.ndarray] = None,
        n_neighbors_effective: int = 0,
    ):
        self.base_model = base_model
        self.mode = str(mode)
        self.group_by = str(group_by)
        self.group_effects = group_effects or {}
        self.sigma_b2 = float(sigma_b2)
        self.sigma_e2 = float(sigma_e2)

        self._nn = knn_model
        self._knn_mode = knn_mode
        self._knn_metric = str(knn_metric)
        self._knn_eps = float(knn_eps)
        self._knn_weighting = str(knn_weighting)
        self._knn_gaussian_sigma = knn_gaussian_sigma
        self._knn_residuals_train = (
            None
            if knn_residuals_train is None
            else np.asarray(knn_residuals_train, dtype=float).reshape(-1)
        )
        self._knn_neighbor_features_train = (
            None
            if knn_neighbor_features_train is None
            else np.asarray(knn_neighbor_features_train, dtype=float)
        )
        self._n_neighbors_effective = int(n_neighbors_effective)

    def predict(self, X, groups=None, neighbor_features=None):
        _, _, combined = self.predict_components(
            X,
            groups=groups,
            neighbor_features=neighbor_features,
        )
        return combined

    def predict_components(self, X, groups=None, neighbor_features=None):
        X_arr = np.asarray(X)
        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)

        if self.mode == "group":
            if groups is None:
                re = np.zeros_like(fixed)
                return fixed, re, fixed + re

            groups_arr = np.asarray(groups).astype(str)
            re = np.array(
                [self.group_effects.get(g, 0.0) for g in groups_arr], dtype=float
            )
            return fixed, re, fixed + re

        if self.mode == "knn":
            if self._nn is None or self._knn_residuals_train is None:
                re = np.zeros_like(fixed)
                return fixed, re, fixed + re

            query_source = neighbor_features if neighbor_features is not None else groups
            if query_source is None:
                re = np.zeros_like(fixed)
                return fixed, re, fixed + re

            query_features = np.asarray(query_source, dtype=float)
            distances, indices = self._nn.kneighbors(
                query_features, return_distance=True
            )
            weights = _compute_knn_weights(
                distances,
                weighting=self._knn_weighting,
                eps=self._knn_eps,
                gaussian_sigma=self._knn_gaussian_sigma,
            )
            numer = np.sum(weights * self._knn_residuals_train[indices], axis=1)
            denom = np.maximum(np.sum(weights, axis=1), self._knn_eps)
            weighted_mean = numer / denom
            n_eff = (denom**2) / np.maximum(np.sum(weights**2, axis=1), self._knn_eps)
            shrink = (n_eff * self.sigma_b2) / np.maximum(
                self.sigma_e2 + n_eff * self.sigma_b2,
                self._knn_eps,
            )
            re = shrink * weighted_mean
            return fixed, re, fixed + re

        re = np.zeros_like(fixed)
        return fixed, re, fixed + re

    def get_group_effects_df(self):
        if self.mode != "group":
            return pd.DataFrame(columns=[self.group_by, "effect"])

        df = pd.DataFrame(
            list(self.group_effects.items()),
            columns=[self.group_by, "effect"],
        )
        if len(df) == 0:
            return df
        return df.sort_values("effect", key=abs, ascending=False)
