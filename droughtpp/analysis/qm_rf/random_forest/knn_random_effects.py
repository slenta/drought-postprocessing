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
    ):
        mode = str(mode).lower()

        self.base_model = base_model
        self.mode = mode
        self.k = int(k)
        self.metric = str(metric)
        self.eps = float(eps)

        self._nn: _t.Optional[NearestNeighbors] = None
        self._residuals_train: _t.Optional[np.ndarray] = None
        self._n_neighbors_effective: _t.Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        neighbor_features_train: np.ndarray,
    ):
        X_arr = np.asarray(X_train)
        y_arr = np.asarray(y_train, dtype=float).reshape(-1)
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

        weights = 1.0 / (distances + self.eps)
        numer = np.sum(weights * self._residuals_train[indices], axis=1)
        denom = np.sum(weights, axis=1)
        return numer / denom

    def predict(self, X, neighbor_features):
        X_arr = np.asarray(X)
        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)
        random_effect = self._predict_random_effect(neighbor_features)
        return fixed + random_effect

    def predict_components(self, X, neighbor_features):
        X_arr = np.asarray(X)
        fixed = np.asarray(self.base_model.predict(X_arr), dtype=float).reshape(-1)
        random_effect = self._predict_random_effect(neighbor_features)
        combined = fixed + random_effect
        return fixed, random_effect, combined
