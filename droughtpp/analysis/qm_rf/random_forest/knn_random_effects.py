import typing as _t

import numpy as np
from sklearn.neighbors import NearestNeighbors


class LocalKNNRandomEffectsRegressor:
    """
    Distance-weighted local KNN random-effects model.

    This wraps a trained base model and adds a local random effect estimated from
    nearest-neighbor residuals in a chosen similarity space.
    """

    def __init__(
        self,
        base_model,
        mode: str,
        k: int = 15,
        metric: str = "euclidean",
        eps: float = 1e-8,
        weighting: str = "inverse_distance",
        gaussian_sigma: _t.Optional[float] = None,
        random_effect_shrinkage: float = 1.0,
    ):
        mode = str(mode).lower()
        weighting_norm = str(weighting).lower()
        aliases = {
            "inverse": "inverse_distance",
            "inverse_distance": "inverse_distance",
            "exp": "exponential",
            "exponential": "exponential",
            "gauss": "gaussian",
            "gaussian": "gaussian",
        }

        self.base_model = base_model
        self.mode = mode
        self.k = int(k)
        self.metric = str(metric)
        self.eps = float(eps)
        self.weighting = aliases.get(weighting_norm, "inverse_distance")
        self.gaussian_sigma = (
            None if gaussian_sigma is None else max(float(gaussian_sigma), self.eps)
        )
        self.random_effect_shrinkage = float(random_effect_shrinkage)

        self._nn: _t.Optional[NearestNeighbors] = None
        self._residuals_train: _t.Optional[np.ndarray] = None
        self._n_neighbors_effective: _t.Optional[int] = None

    def _proximity_leaf_features(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X)
        if hasattr(self.base_model, "apply"):
            leaves = self.base_model.apply(X_arr)
        elif hasattr(self.base_model, "get_booster"):
            import xgboost as xgb

            booster = self.base_model.get_booster()
            leaves = booster.predict(xgb.DMatrix(X_arr), pred_leaf=True)

        leaves = np.asarray(leaves, dtype=float)
        if leaves.ndim == 1:
            leaves = leaves.reshape(-1, 1)
        return leaves

    def _compute_weights(self, distances: np.ndarray) -> np.ndarray:
        if self.weighting == "inverse_distance":
            return 1.0 / (distances + self.eps)

        if self.weighting == "exponential":
            return np.exp(-distances)

        if self.weighting == "gaussian":
            if self.gaussian_sigma is None:
                sigma = np.mean(distances, axis=1, keepdims=True)
                sigma = np.maximum(sigma, self.eps)
            else:
                sigma = self.gaussian_sigma
            return np.exp(-(distances**2) / (2.0 * (sigma**2)))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        neighbor_features_train: _t.Optional[np.ndarray],
    ):
        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train, dtype=float).reshape(-1)
        if self.mode == "proximity":
            nfeat = self._proximity_leaf_features(X_arr)
        else:
            nfeat = np.asarray(neighbor_features_train, dtype=float)

        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)

        self._residuals_train = y_arr - fixed
        self._n_neighbors_effective = min(self.k, nfeat.shape[0])

        nn = NearestNeighbors(
            n_neighbors=self._n_neighbors_effective,
            metric=self.metric,
        )
        nn.fit(nfeat)
        self._nn = nn
        return self

    def _predict_random_effect(self, neighbor_features: np.ndarray) -> np.ndarray:

        q = np.asarray(neighbor_features, dtype=float)
        if q.shape[0] == 0:
            return np.array([], dtype=float)

        distances, indices = self._nn.kneighbors(q, return_distance=True)

        weights = self._compute_weights(distances)
        numer = np.sum(weights * self._residuals_train[indices], axis=1)
        denom = np.maximum(np.sum(weights, axis=1), self.eps)
        return numer / denom

    def predict(self, X, neighbor_features=None):
        X_arr = np.asarray(X)
        if self.mode == "proximity":
            query_features = self._proximity_leaf_features(X_arr)
        else:
            query_features = neighbor_features
        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)
        random_effect = self.random_effect_shrinkage * self._predict_random_effect(
            query_features
        )
        return fixed + random_effect

    def predict_components(self, X, neighbor_features=None):
        X_arr = np.asarray(X)
        if self.mode == "proximity":
            query_features = self._proximity_leaf_features(X_arr)
        else:
            query_features = neighbor_features
        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)
        random_effect = self.random_effect_shrinkage * self._predict_random_effect(
            query_features
        )
        combined = fixed + random_effect
        return fixed, random_effect, combined
