import typing as _t
import numpy as np
import xarray as xr
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from .knn_random_effects import LocalKNNRandomEffectsRegressor
from .BLUP_random_effects import (
    BLUPMixedEffectsRegressor,
    compute_group_blup_random_effects,
    compute_knn_blup_random_effects,
    estimate_variance_components_iterative,
)


class MixedEffectsTreeRegressor:
    def __init__(
        self,
        base_model,
        group_effects: dict[str, float],
        group_by: str,
        random_effect_shrinkage: float = 1.0,
    ):
        self.base_model = base_model
        self.group_effects = group_effects
        self.group_by = group_by
        self.random_effect_shrinkage = float(random_effect_shrinkage)

    def predict(self, X, groups):
        groups_arr = np.asarray(groups).astype(str)
        fixed = self.base_model.predict(X)
        re = self.random_effect_shrinkage * np.array(
            [self.group_effects.get(g, 0.0) for g in groups_arr], dtype=float
        )
        return fixed + re

    def predict_components(self, X, groups):
        """
        Predict and return both fixed (base) and random (group) effects separately.
        Returns tuple: (fixed_effects, random_effects, combined_prediction)
        """
        groups_arr = np.asarray(groups).astype(str)
        fixed = self.base_model.predict(X)
        re = self.random_effect_shrinkage * np.array(
            [self.group_effects.get(g, 0.0) for g in groups_arr], dtype=float
        )
        combined = fixed + re
        return fixed, re, combined

    def get_group_effects_df(self):
        """Return group effects as a sorted DataFrame."""
        df = pd.DataFrame(
            list(self.group_effects.items()), columns=[self.group_by, "effect"]
        )
        df = df.sort_values("effect", key=abs, ascending=False)
        return df


class RandomForestBiasCorrector:
    """
    Simple RF-based bias corrector for quantile-mapping residuals.

    Usage:
      rf = RandomForestBiasCorrector()
      X, y, stacked_ref = rf.prepare_training_data(da_res, predictor_ds, predictor_vars)
      rf.train(X, y)
    corrected, predicted = rf.apply_rf_correction(qm_da, predictor_ds, predictor_vars)
      rf.save_corrected_and_predicted(corrected, predicted, corrected_path, residual_path, var_name=var)
    """

    def __init__(self):
        self.model: _t.Optional[
            _t.Union[
                RandomForestRegressor,
                XGBRegressor,
                MixedEffectsTreeRegressor,
                LocalKNNRandomEffectsRegressor,
                BLUPMixedEffectsRegressor,
            ]
        ] = None
        self.training_history: _t.Optional[dict] = None

    @staticmethod
    def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

    def _train_xgboost(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: _t.Optional[pd.DataFrame],
        y_val: _t.Optional[pd.Series],
        n_estimators: int,
        random_state: int,
        xgb_learning_rate: float,
        xgb_max_depth: int,
        xgb_subsample: float,
        xgb_colsample_bytree: float,
        xgb_min_child_weight: float,
        xgb_eval_metric: str,
        xgb_early_stopping_rounds: _t.Optional[int] = None,
    ) -> XGBRegressor:
        use_early_stopping = (
            xgb_early_stopping_rounds is not None
            and int(xgb_early_stopping_rounds) > 0
            and X_val is not None
            and y_val is not None
        )

        xgb_kwargs = {
            "n_estimators": n_estimators,
            "random_state": random_state,
            "learning_rate": xgb_learning_rate,
            "max_depth": xgb_max_depth,
            "subsample": xgb_subsample,
            "colsample_bytree": xgb_colsample_bytree,
            "min_child_weight": xgb_min_child_weight,
            "eval_metric": xgb_eval_metric,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        if use_early_stopping:
            xgb_kwargs["early_stopping_rounds"] = int(xgb_early_stopping_rounds)

        xgb = XGBRegressor(**xgb_kwargs)
        if X_val is not None and y_val is not None:
            eval_set = [(X.values, y.values), (X_val.values, y_val.values)]
        else:
            eval_set = [(X.values, y.values)]

        xgb.fit(
            X.values,
            y.values,
            eval_set=eval_set,
            verbose=False,
        )

        evals_result = xgb.evals_result()
        train_key = "validation_0"
        val_key = "validation_1"
        train_hist = evals_result.get(train_key, {}).get(xgb_eval_metric, [])
        val_hist = evals_result.get(val_key, {}).get(xgb_eval_metric, [])
        n_rounds = list(range(1, len(train_hist) + 1))
        self.training_history = {
            "model_type": "xgboost",
            "metric": xgb_eval_metric,
            "x": n_rounds,
            "train": train_hist,
            "val": val_hist if len(val_hist) > 0 else None,
        }
        return xgb

    def _train_rf(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: _t.Optional[pd.DataFrame],
        y_val: _t.Optional[pd.Series],
        n_estimators: int,
        chunk_size: int,
        random_state: int,
        rf_max_depth: _t.Optional[int],
        rf_min_samples_leaf: int,
        rf_max_features: _t.Union[str, int, float, None],
        rf_early_stopping_patience: int = 0,
        rf_early_stopping_min_delta: float = 0.0,
    ) -> RandomForestRegressor:
        start = time.time()
        total_done = 0
        first = min(chunk_size, n_estimators)

        rf = RandomForestRegressor(
            n_estimators=first,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
        )
        rf.fit(X.values, y.values)
        total_done += first

        train_rmse = [float(np.sqrt(np.mean((rf.predict(X.values) - y.values) ** 2)))]
        val_rmse = []
        best_val_rmse = np.inf
        best_n_estimators = total_done
        no_improve = 0
        if X_val is not None and y_val is not None:
            first_val = self._rmse(y_val.values, rf.predict(X_val.values))
            val_rmse.append(first_val)
            best_val_rmse = first_val
        trees_hist = [total_done]

        pbar = tqdm(total=n_estimators, desc="RF training", unit="trees")
        pbar.update(first)

        while total_done < n_estimators:
            chunk_start = time.time()
            add = min(chunk_size, n_estimators - total_done)
            rf.n_estimators = total_done + add
            rf.fit(X.values, y.values)
            total_done += add
            chunk_time = time.time() - chunk_start

            elapsed = time.time() - start
            avg_per_tree = elapsed / total_done
            remaining = n_estimators - total_done
            eta = remaining * avg_per_tree

            pbar.update(add)
            print(
                f"Elapsed {elapsed:.0f}s — trees {total_done}/{n_estimators} — last_chunk {chunk_time:.1f}s — ETA {eta:.0f}s"
            )

            trees_hist.append(total_done)
            train_rmse.append(self._rmse(y.values, rf.predict(X.values)))
            if X_val is not None and y_val is not None:
                current_val = self._rmse(y_val.values, rf.predict(X_val.values))
                val_rmse.append(current_val)

                improved = (best_val_rmse - current_val) > float(
                    rf_early_stopping_min_delta
                )
                if improved:
                    best_val_rmse = current_val
                    best_n_estimators = total_done
                    no_improve = 0
                else:
                    no_improve += 1

                if int(rf_early_stopping_patience) > 0 and no_improve >= int(
                    rf_early_stopping_patience
                ):
                    break

        pbar.close()

        if (
            X_val is not None
            and y_val is not None
            and int(rf_early_stopping_patience) > 0
            and best_n_estimators < total_done
        ):
            rf = RandomForestRegressor(
                n_estimators=best_n_estimators,
                n_jobs=-1,
                random_state=random_state,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
                max_features=rf_max_features,
            )
            rf.fit(X.values, y.values)

            cutoff = max(1, len([n for n in trees_hist if n <= best_n_estimators]))
            trees_hist = trees_hist[:cutoff]
            train_rmse = train_rmse[:cutoff]
            val_rmse = val_rmse[:cutoff]

        self.training_history = {
            "model_type": "rf",
            "metric": "rmse",
            "x": trees_hist,
            "train": train_rmse,
            "val": val_rmse if len(val_rmse) > 0 else None,
        }
        return rf

    @staticmethod
    def _stack_by_samples(da: xr.DataArray) -> xr.DataArray:
        """Stack all dims into a single 'sample' dimension for model training/prediction."""
        sample_dims = list(da.dims)
        return da.stack(sample=sample_dims)

    def prepare_training_data(
        self,
        residual_da: xr.DataArray,
        predictor_ds: xr.Dataset,
        predictor_vars: _t.List[str],
    ) -> _t.Tuple[pd.DataFrame, pd.Series, xr.DataArray]:
        """
        Prepare flattened training data.
        Returns (X_df, y_series, stacked_reference_da) where stacked_reference_da holds coords for unstacking.
        """
        preds = [predictor_ds[var] for var in predictor_vars]
        aligned = xr.align(residual_da, *preds, join="exact")
        res_aligned = aligned[0]
        preds_aligned = aligned[1:]

        res_s = self._stack_by_samples(res_aligned)
        pred_s_list = [self._stack_by_samples(p) for p in preds_aligned]

        data = {}
        for name, p in zip(predictor_vars, pred_s_list):
            data[name] = p.values
        X = pd.DataFrame(data)
        y = pd.Series(res_s.values, name="residual")

        mask = np.isfinite(y.values)
        finite_cols = np.all(np.isfinite(X.values), axis=1)
        good = mask & finite_cols
        X = X.loc[good].reset_index(drop=True)
        y = y.loc[good].reset_index(drop=True)

        return X, y, res_s

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: _t.Optional[pd.DataFrame] = None,
        y_val: _t.Optional[pd.Series] = None,
        groups: _t.Optional[_t.Sequence[_t.Any]] = None,
        groups_val: _t.Optional[_t.Sequence[_t.Any]] = None,
        model_type: str = "rf",
        mixed_base_model: str = "rf",
        mixed_group_by: str = "grid_id",
        n_estimators: int = 100,
        chunk_size: int = 10,
        random_state: int = 0,
        xgb_learning_rate: float = 0.1,
        xgb_max_depth: int = 6,
        xgb_subsample: float = 0.8,
        xgb_colsample_bytree: float = 0.8,
        xgb_min_child_weight: float = 1.0,
        xgb_eval_metric: str = "rmse",
        rf_max_depth: _t.Optional[int] = None,
        rf_min_samples_leaf: int = 1,
        rf_max_features: _t.Union[str, int, float, None] = "sqrt",
        knn_neighbor_features: _t.Optional[np.ndarray] = None,
        knn_neighbor_features_val: _t.Optional[np.ndarray] = None,
        knn_k: int = 15,
        knn_metric: str = "euclidean",
        knn_eps: float = 1e-8,
        knn_weighting: str = "inverse_distance",
        knn_gaussian_sigma: _t.Optional[float] = None,
        re_shrinkage_grid: _t.Optional[_t.Sequence[float]] = None,
        tune_knn_k_values: _t.Optional[_t.Sequence[int]] = None,
        tune_knn_eps_values: _t.Optional[_t.Sequence[float]] = None,
        merf_n_iter: int = 1,
        merf_tol: float = 1e-6,
        mixed_effects_impl: str = "blup",
        mixed_likelihood_target: str = "reml",
        variance_max_iter: int = 25,
        variance_tol: float = 1e-6,
        variance_floor: float = 1e-8,
        xgb_early_stopping_rounds: _t.Optional[int] = None,
        rf_early_stopping_patience: int = 0,
        rf_early_stopping_min_delta: float = 0.0,
    ) -> _t.Union[
        RandomForestRegressor,
        XGBRegressor,
        MixedEffectsTreeRegressor,
        LocalKNNRandomEffectsRegressor,
        BLUPMixedEffectsRegressor,
    ]:

        model_type = str(model_type).lower()
        if model_type in {"xgb", "xgboost"}:
            model = self._train_xgboost(
                X,
                y,
                X_val,
                y_val,
                n_estimators,
                random_state,
                xgb_learning_rate,
                xgb_max_depth,
                xgb_subsample,
                xgb_colsample_bytree,
                xgb_min_child_weight,
                xgb_eval_metric,
                xgb_early_stopping_rounds,
            )
            self.model = model
            return model

        if model_type in {"mixed_rf", "mixed"}:

            mixed_base = str(mixed_base_model).lower()
            mixed_impl = str(mixed_effects_impl).lower().strip()
            shrinkage_candidates = (
                [float(v) for v in re_shrinkage_grid]
                if re_shrinkage_grid is not None and len(re_shrinkage_grid) > 0
                else [1.0]
            )

            group_mode = str(mixed_group_by).lower()
            if group_mode in {"knn_spatial", "knn_feature", "knn_proximity"}:
                n_iter = max(1, int(merf_n_iter))
                tol = float(merf_tol)
                random_effects = np.zeros(len(y), dtype=float)
                sigma_b2 = 1.0
                sigma_e2 = 1.0
                if group_mode == "knn_spatial":
                    knn_mode = "spatial"
                elif group_mode == "knn_feature":
                    knn_mode = "feature"
                else:
                    knn_mode = "proximity"
                n_iter_done = 0
                knn_features_arr = np.asarray(knn_neighbor_features)
                knn_features_val_arr = (
                    np.asarray(knn_neighbor_features_val)
                    if knn_neighbor_features_val is not None
                    else None
                )
                k_candidates = (
                    [int(v) for v in tune_knn_k_values]
                    if tune_knn_k_values is not None and len(tune_knn_k_values) > 0
                    else [int(knn_k)]
                )
                eps_candidates = (
                    [float(v) for v in tune_knn_eps_values]
                    if tune_knn_eps_values is not None and len(tune_knn_eps_values) > 0
                    else [float(knn_eps)]
                )
                iteration_diagnostics = []
                best_knn_cfg: dict[str, _t.Any] = {
                    "k": int(knn_k),
                    "eps": float(knn_eps),
                    "shrinkage": float(shrinkage_candidates[0]),
                }
                knn_model = None

                for _ in range(n_iter):
                    n_iter_done += 1
                    y_adjusted = pd.Series(
                        y.values - random_effects,
                        index=y.index,
                        name=y.name,
                    )

                    if mixed_base in {"xgb", "xgboost"}:
                        base_model = self._train_xgboost(
                            X,
                            y_adjusted,
                            X_val,
                            y_val,
                            n_estimators,
                            random_state,
                            xgb_learning_rate,
                            xgb_max_depth,
                            xgb_subsample,
                            xgb_colsample_bytree,
                            xgb_min_child_weight,
                            xgb_eval_metric,
                            xgb_early_stopping_rounds,
                        )
                    else:
                        base_model = self._train_rf(
                            X,
                            y_adjusted,
                            X_val,
                            y_val,
                            n_estimators,
                            chunk_size,
                            random_state,
                            rf_max_depth,
                            rf_min_samples_leaf,
                            rf_max_features,
                            rf_early_stopping_patience,
                            rf_early_stopping_min_delta,
                        )

                    fixed_train = np.asarray(base_model.predict(X.values), dtype=float)
                    residuals_train = np.asarray(y.values - fixed_train, dtype=float)

                    best_score = np.inf
                    best_random_effects = None
                    best_candidate_nn = None
                    best_n_neighbors = int(knn_k)
                    best_sigma_b2 = float(sigma_b2)
                    best_sigma_e2 = float(sigma_e2)
                    best_ll = float("-inf")
                    best_basic_model = None
                    fixed_val = None
                    blup_candidate_cache = {}
                    if mixed_impl != "basic":
                        fixed_val = np.asarray(
                            base_model.predict(X_val.values), dtype=float
                        )
                    for alpha in shrinkage_candidates:
                        for cand_k in k_candidates:
                            for cand_eps in eps_candidates:
                                if mixed_impl == "basic":
                                    candidate_model = LocalKNNRandomEffectsRegressor(
                                        base_model=base_model,
                                        mode=knn_mode,
                                        k=int(cand_k),
                                        metric=knn_metric,
                                        eps=float(cand_eps),
                                        weighting=knn_weighting,
                                        gaussian_sigma=knn_gaussian_sigma,
                                        random_effect_shrinkage=float(alpha),
                                    )
                                    candidate_model.fit(
                                        X_train=X.values,
                                        y_train=y.values,
                                        neighbor_features_train=knn_features_arr,
                                    )

                                    val_pred = candidate_model.predict(
                                        X_val.values,
                                        knn_features_val_arr,
                                    )
                                    rmse_score = self._rmse(y_val.values, val_pred)

                                    score = rmse_score
                                    ll_score = float("nan")
                                    sigma_b2_new = float(sigma_b2)
                                    sigma_e2_new = float(sigma_e2)
                                    candidate_nn = getattr(candidate_model, "_nn", None)
                                    n_neighbors_eff = int(
                                        getattr(
                                            candidate_model,
                                            "_n_neighbors_effective",
                                            cand_k,
                                        )
                                    )
                                else:
                                    cache_key = (int(cand_k), float(cand_eps))
                                    cache_entry = blup_candidate_cache.get(cache_key)
                                    if cache_entry is None:
                                        re_train_blup, candidate_nn, n_neighbors_eff = (
                                            compute_knn_blup_random_effects(
                                                residuals_train=residuals_train,
                                                neighbor_features_train=knn_features_arr,
                                                neighbor_features_query=knn_features_arr,
                                                k=int(cand_k),
                                                metric=knn_metric,
                                                eps=float(cand_eps),
                                                weighting=knn_weighting,
                                                sigma_b2=max(
                                                    float(sigma_b2),
                                                    float(variance_floor),
                                                ),
                                                sigma_e2=max(
                                                    float(sigma_e2),
                                                    float(variance_floor),
                                                ),
                                                gaussian_sigma=knn_gaussian_sigma,
                                            )
                                        )
                                        re_val_blup, _, _ = (
                                            compute_knn_blup_random_effects(
                                                residuals_train=residuals_train,
                                                neighbor_features_train=knn_features_arr,
                                                neighbor_features_query=knn_features_val_arr,
                                                k=int(cand_k),
                                                metric=knn_metric,
                                                eps=float(cand_eps),
                                                weighting=knn_weighting,
                                                sigma_b2=max(
                                                    float(sigma_b2),
                                                    float(variance_floor),
                                                ),
                                                sigma_e2=max(
                                                    float(sigma_e2),
                                                    float(variance_floor),
                                                ),
                                                gaussian_sigma=knn_gaussian_sigma,
                                            )
                                        )
                                        cache_entry = (
                                            re_train_blup,
                                            re_val_blup,
                                            candidate_nn,
                                            int(n_neighbors_eff),
                                        )
                                        blup_candidate_cache[cache_key] = cache_entry
                                    (
                                        re_train_blup,
                                        re_val_blup,
                                        candidate_nn,
                                        n_neighbors_eff,
                                    ) = cache_entry
                                    re_train_alpha = float(alpha) * re_train_blup
                                    sigma_b2_new = float(sigma_b2)
                                    sigma_e2_new = float(sigma_e2)
                                    ll_score = float("nan")

                                    val_pred = fixed_val + float(alpha) * re_val_blup
                                    rmse_score = self._rmse(y_val.values, val_pred)

                                    score = rmse_score

                                if score < best_score:
                                    best_score = score
                                    if mixed_impl == "basic":
                                        best_random_effects = None
                                        best_basic_model = candidate_model
                                    else:
                                        best_random_effects = re_train_alpha
                                    best_candidate_nn = candidate_nn
                                    best_n_neighbors = int(n_neighbors_eff)
                                    best_sigma_b2 = float(sigma_b2_new)
                                    best_sigma_e2 = float(sigma_e2_new)
                                    best_ll = float(ll_score)
                                    best_knn_cfg = {
                                        "k": int(cand_k),
                                        "eps": float(cand_eps),
                                        "shrinkage": float(alpha),
                                    }

                    if mixed_impl == "basic":
                        knn_model = best_basic_model
                        if knn_mode == "proximity":
                            _, updated_random_effects, _ = knn_model.predict_components(
                                X.values,
                                None,
                            )
                        else:
                            _, updated_random_effects, _ = knn_model.predict_components(
                                X.values,
                                knn_features_arr,
                            )
                    else:
                        updated_random_effects = np.asarray(
                            best_random_effects, dtype=float
                        )
                        sigma_b2, sigma_e2, var_hist = (
                            estimate_variance_components_iterative(
                                residuals=residuals_train,
                                random_effects=updated_random_effects,
                                objective=mixed_likelihood_target,
                                n_fixed_effects=int(X.shape[1]),
                                sigma_b2_init=max(
                                    float(sigma_b2), float(variance_floor)
                                ),
                                sigma_e2_init=max(
                                    float(sigma_e2), float(variance_floor)
                                ),
                                max_iter=int(variance_max_iter),
                                tol=float(variance_tol),
                                var_floor=float(variance_floor),
                                groups=None,
                                is_knn=True,
                            )
                        )
                        best_ll = (
                            float(var_hist[-1]["log_likelihood"])
                            if len(var_hist) > 0
                            else float("nan")
                        )
                        knn_model = BLUPMixedEffectsRegressor(
                            base_model=base_model,
                            mode="knn",
                            group_by=mixed_group_by,
                            sigma_b2=sigma_b2,
                            sigma_e2=sigma_e2,
                            knn_model=best_candidate_nn,
                            knn_mode=knn_mode,
                            knn_metric=knn_metric,
                            knn_eps=float(best_knn_cfg["eps"]),
                            knn_weighting=knn_weighting,
                            knn_gaussian_sigma=knn_gaussian_sigma,
                            knn_residuals_train=residuals_train,
                            knn_neighbor_features_train=knn_features_arr,
                            n_neighbors_effective=best_n_neighbors,
                        )

                    iter_diag = analyze_merf_contributions(
                        knn_model,
                        X.values,
                        knn_features_arr,
                        y.values,
                    )
                    if iter_diag is not None:
                        iter_diag["iteration"] = n_iter_done
                        iter_diag["val_rmse"] = float(best_score)
                        if mixed_impl == "basic":
                            iter_diag["objective"] = "rmse"
                        else:
                            iter_diag["objective"] = str(
                                mixed_likelihood_target
                            ).lower()
                        iter_diag["log_likelihood"] = float(best_ll)
                        iter_diag["random_effect_shrinkage"] = float(
                            best_knn_cfg["shrinkage"]
                        )
                        iter_diag["sigma_b2"] = float(sigma_b2)
                        iter_diag["sigma_e2"] = float(sigma_e2)
                        iter_diag["knn_k"] = int(best_knn_cfg["k"])
                        iter_diag["knn_eps"] = float(best_knn_cfg["eps"])
                        iter_diag.pop("base_predictions", None)
                        iter_diag.pop("group_effects_predictions", None)
                        iter_diag.pop("combined_predictions", None)
                        iteration_diagnostics.append(iter_diag)

                    max_delta = (
                        float(np.max(np.abs(updated_random_effects - random_effects)))
                        if len(updated_random_effects) > 0
                        else 0.0
                    )
                    random_effects = updated_random_effects
                    if max_delta <= tol:
                        break

                self.training_history["model_type"] = f"mixed_knn[{mixed_base}]"
                self.training_history["mixed_effects_impl"] = mixed_impl
                self.training_history["merf_n_iter"] = n_iter_done
                self.training_history["merf_tol"] = tol
                self.training_history["random_effect_shrinkage"] = float(
                    best_knn_cfg["shrinkage"]
                )
                self.training_history["mixed_likelihood_target"] = str(
                    mixed_likelihood_target
                ).lower()
                self.training_history["sigma_b2"] = float(sigma_b2)
                self.training_history["sigma_e2"] = float(sigma_e2)
                self.training_history["knn_k"] = int(best_knn_cfg["k"])
                self.training_history["knn_eps"] = float(best_knn_cfg["eps"])
                self.training_history["merf_iteration_diagnostics"] = (
                    iteration_diagnostics
                )
                self.model = knn_model
                return knn_model

            groups_arr = np.asarray(groups).astype(str)
            groups_val_arr = (
                np.asarray(groups_val).astype(str) if groups_val is not None else None
            )
            n_iter = max(1, int(merf_n_iter))
            tol = float(merf_tol)
            random_effects = np.zeros(len(y), dtype=float)
            group_effects: dict[str, float] = {}
            best_shrinkage = float(shrinkage_candidates[0])
            sigma_b2 = 1.0
            sigma_e2 = 1.0
            n_iter_done = 0
            iteration_diagnostics = []
            best_ll = float("-inf")

            for _ in range(n_iter):
                n_iter_done += 1
                y_adjusted = pd.Series(
                    y.values - random_effects,
                    index=y.index,
                    name=y.name,
                )

                if mixed_base in {"xgb", "xgboost"}:
                    base_model = self._train_xgboost(
                        X,
                        y_adjusted,
                        X_val,
                        y_val,
                        n_estimators,
                        random_state,
                        xgb_learning_rate,
                        xgb_max_depth,
                        xgb_subsample,
                        xgb_colsample_bytree,
                        xgb_min_child_weight,
                        xgb_eval_metric,
                        xgb_early_stopping_rounds,
                    )
                else:
                    base_model = self._train_rf(
                        X,
                        y_adjusted,
                        X_val,
                        y_val,
                        n_estimators,
                        chunk_size,
                        random_state,
                        rf_max_depth,
                        rf_min_samples_leaf,
                        rf_max_features,
                        rf_early_stopping_patience,
                        rf_early_stopping_min_delta,
                    )

                fixed_preds = base_model.predict(X.values)
                residuals = y.values - fixed_preds
                if mixed_impl == "basic":
                    group_effects_series = (
                        pd.Series(residuals).groupby(groups_arr).mean()
                    )
                    group_effects_raw = {
                        str(group): float(effect)
                        for group, effect in group_effects_series.items()
                    }
                    re_blup = np.asarray(
                        [group_effects_raw.get(g, 0.0) for g in groups_arr],
                        dtype=float,
                    )
                else:
                    group_effects_raw, re_blup = compute_group_blup_random_effects(
                        residuals=residuals,
                        groups=groups_arr,
                        sigma_b2=max(float(sigma_b2), float(variance_floor)),
                        sigma_e2=max(float(sigma_e2), float(variance_floor)),
                    )

                best_score = np.inf
                for alpha in shrinkage_candidates:
                    re_candidate = float(alpha) * re_blup
                    if mixed_impl == "basic":
                        sigma_b2_new = float(sigma_b2)
                        sigma_e2_new = float(sigma_e2)
                        ll_score = float("nan")
                    else:
                        sigma_b2_new, sigma_e2_new, var_hist = (
                            estimate_variance_components_iterative(
                                residuals=residuals,
                                random_effects=re_candidate,
                                objective=mixed_likelihood_target,
                                n_fixed_effects=int(X.shape[1]),
                                sigma_b2_init=max(
                                    float(sigma_b2), float(variance_floor)
                                ),
                                sigma_e2_init=max(
                                    float(sigma_e2), float(variance_floor)
                                ),
                                max_iter=int(variance_max_iter),
                                tol=float(variance_tol),
                                var_floor=float(variance_floor),
                                groups=groups_arr,
                                is_knn=False,
                            )
                        )
                        ll_score = (
                            float(var_hist[-1]["log_likelihood"])
                            if len(var_hist) > 0
                            else float("-inf")
                        )

                    val_re = float(alpha) * np.array(
                        [group_effects_raw.get(g, 0.0) for g in groups_val_arr],
                        dtype=float,
                    )
                    rmse_score = self._rmse(
                        y_val.values,
                        base_model.predict(X_val.values) + val_re,
                    )

                    if mixed_impl == "basic":
                        score = rmse_score
                    else:
                        objective_score = -ll_score
                        if not np.isfinite(objective_score):
                            objective_score = rmse_score
                        score = objective_score
                    if score < best_score:
                        best_score = score
                        best_shrinkage = float(alpha)
                        best_ll = float(ll_score)
                        sigma_b2 = float(sigma_b2_new)
                        sigma_e2 = float(sigma_e2_new)
                        group_effects = {
                            str(group): float(best_shrinkage * effect)
                            for group, effect in group_effects_raw.items()
                        }

                updated_random_effects = np.array(
                    [group_effects.get(g, 0.0) for g in groups_arr],
                    dtype=float,
                )
                if mixed_impl == "basic":
                    iter_model = MixedEffectsTreeRegressor(
                        base_model=base_model,
                        group_effects=group_effects,
                        group_by=mixed_group_by,
                        random_effect_shrinkage=1.0,
                    )
                else:
                    iter_model = BLUPMixedEffectsRegressor(
                        base_model=base_model,
                        mode="group",
                        group_effects=group_effects,
                        group_by=mixed_group_by,
                        sigma_b2=sigma_b2,
                        sigma_e2=sigma_e2,
                    )
                iter_diag = analyze_merf_contributions(
                    iter_model,
                    X.values,
                    groups_arr,
                    y.values,
                )
                if iter_diag is not None:
                    iter_diag["iteration"] = n_iter_done
                    iter_diag["val_rmse"] = float(best_score)
                    if mixed_impl == "basic":
                        iter_diag["objective"] = "rmse"
                    else:
                        iter_diag["objective"] = str(mixed_likelihood_target).lower()
                    iter_diag["log_likelihood"] = float(best_ll)
                    iter_diag["random_effect_shrinkage"] = float(best_shrinkage)
                    iter_diag["sigma_b2"] = float(sigma_b2)
                    iter_diag["sigma_e2"] = float(sigma_e2)
                    iter_diag.pop("base_predictions", None)
                    iter_diag.pop("group_effects_predictions", None)
                    iter_diag.pop("combined_predictions", None)
                    iteration_diagnostics.append(iter_diag)

                max_delta = (
                    float(np.max(np.abs(updated_random_effects - random_effects)))
                    if len(updated_random_effects) > 0
                    else 0.0
                )
                random_effects = updated_random_effects
                if max_delta <= tol:
                    break

            if mixed_impl == "basic":
                mixed_model = MixedEffectsTreeRegressor(
                    base_model=base_model,
                    group_effects=group_effects,
                    group_by=mixed_group_by,
                    random_effect_shrinkage=1.0,
                )
            else:
                mixed_model = BLUPMixedEffectsRegressor(
                    base_model=base_model,
                    mode="group",
                    group_effects=group_effects,
                    group_by=mixed_group_by,
                    sigma_b2=sigma_b2,
                    sigma_e2=sigma_e2,
                )
            self.training_history["model_type"] = f"mixed_rf[{mixed_base}]"
            self.training_history["mixed_effects_impl"] = mixed_impl
            self.training_history["merf_n_iter"] = n_iter_done
            self.training_history["merf_tol"] = tol
            self.training_history["random_effect_shrinkage"] = float(best_shrinkage)
            self.training_history["mixed_likelihood_target"] = str(
                mixed_likelihood_target
            ).lower()
            self.training_history["sigma_b2"] = float(sigma_b2)
            self.training_history["sigma_e2"] = float(sigma_e2)
            self.training_history["merf_iteration_diagnostics"] = iteration_diagnostics
            self.model = mixed_model
            return mixed_model

        model = self._train_rf(
            X,
            y,
            X_val,
            y_val,
            n_estimators,
            chunk_size,
            random_state,
            rf_max_depth,
            rf_min_samples_leaf,
            rf_max_features,
            rf_early_stopping_patience,
            rf_early_stopping_min_delta,
        )
        self.model = model
        return model

    def apply_rf_correction(
        self,
        qm_da: xr.DataArray,
        predictor_ds: xr.Dataset,
        predictor_vars: _t.List[str],
    ) -> _t.Tuple[xr.DataArray, xr.DataArray]:
        """
        Apply trained RF to predict residuals and correct qm_da.
        Returns (corrected_da, predicted_residual_da).
        """

        preds = [predictor_ds[var] for var in predictor_vars]
        aligned = xr.align(qm_da, *preds, join="exact")
        qm_aligned = aligned[0]
        preds_aligned = aligned[1:]

        pred_s_list = [self._stack_by_samples(p) for p in preds_aligned]

        X_pred = pd.DataFrame(
            {name: p.values for name, p in zip(predictor_vars, pred_s_list)}
        )
        finite_mask = np.all(np.isfinite(X_pred.values), axis=1)
        preds_array = np.full(X_pred.shape[0], np.nan, dtype=float)
        if finite_mask.any():
            preds_array[finite_mask] = self.model.predict(X_pred.values[finite_mask])

        pred_da_stacked = xr.DataArray(
            preds_array,
            coords=(pred_s_list[0].coords["sample"],),
            dims=("sample",),
        )
        pred_da = pred_da_stacked.unstack("sample").transpose(*qm_aligned.dims)

        corrected = qm_aligned + pred_da

        return corrected, pred_da

    @staticmethod
    def save_corrected_and_predicted(
        corrected_da: xr.DataArray,
        predicted_residual_da: xr.DataArray,
        corrected_path: str,
        residual_path: str,
        var_name: str = None,
    ):
        """Save corrected variable and predicted residuals to netCDF files."""
        ds_corr = corrected_da.to_dataset(
            name=(var_name or corrected_da.name or "corrected")
        )
        ds_pred = predicted_residual_da.to_dataset(
            name=(var_name or predicted_residual_da.name or "predicted_residual")
        )
        ds_corr.to_netcdf(corrected_path)
        ds_pred.to_netcdf(residual_path)


def analyze_merf_contributions(
    model: _t.Union[
        MixedEffectsTreeRegressor,
        LocalKNNRandomEffectsRegressor,
        BLUPMixedEffectsRegressor,
        XGBRegressor,
    ],
    X: np.ndarray,
    groups: np.ndarray,
    y_true: _t.Optional[np.ndarray] = None,
):
    """
    Analyze MERF model contributions (base term vs. group effects).
    Returns a dictionary with diagnostics and creates a summary table.

    Args:
        model: Trained MixedEffectsTreeRegressor or plain model
        X: Feature matrix
        groups: Group labels for each sample
        y_true: Optional true values for error computation

    Returns:
        dict with keys: group_effects_df, base_stats, random_stats, summary_table, plots (optional)
    """
    if not isinstance(
        model,
        (
            MixedEffectsTreeRegressor,
            LocalKNNRandomEffectsRegressor,
            BLUPMixedEffectsRegressor,
        ),
    ):
        print("Model is not a MERF; skipping component analysis.")
        return None

    # Get component predictions
    fixed, re, combined = model.predict_components(X, groups)

    # MixedEffectsTreeRegressor has explicit group effects, KNN mixed does not.
    if isinstance(model, (MixedEffectsTreeRegressor, BLUPMixedEffectsRegressor)):
        group_effects_df = model.get_group_effects_df()
        group_effects = getattr(model, "group_effects", {})
        n_groups = len(group_effects)
    else:
        group_effects_df = pd.DataFrame(columns=["group", "effect"])
        group_effects = {}
        n_groups = int(getattr(model, "_n_neighbors_effective", 0) or 0)

    # Statistics on fixed term
    base_stats = {
        "mean": float(np.nanmean(fixed)),
        "std": float(np.nanstd(fixed)),
        "min": float(np.nanmin(fixed)),
        "max": float(np.nanmax(fixed)),
        "range": float(np.nanmax(fixed) - np.nanmin(fixed)),
    }

    # Statistics on random effects
    random_stats = {
        "n_groups": n_groups,
        "mean": float(np.nanmean(re)),
        "std": float(np.nanstd(re)),
        "min": float(np.nanmin(re)),
        "max": float(np.nanmax(re)),
        "range": float(np.nanmax(re) - np.nanmin(re)),
        "pct_nonzero": 100.0 * np.sum(np.abs(re) > 1e-10) / len(re),
    }

    # Contribution analysis
    abs_fixed = np.abs(fixed)
    abs_re = np.abs(re)
    eps = 1e-10
    contribution_ratio = abs_re / (abs_fixed + abs_re + eps)

    results = {
        "group_effects": group_effects,
        "group_effects_df": group_effects_df,
        "base_stats": base_stats,
        "random_stats": random_stats,
        "base_predictions": fixed,
        "group_effects_predictions": re,
        "combined_predictions": combined,
        "contribution_ratio_groups": np.nanmean(contribution_ratio),
        "max_contribution_ratio": np.nanmax(contribution_ratio),
    }

    base_rmse = float(np.sqrt(np.nanmean((fixed - y_true) ** 2)))
    combined_rmse = float(np.sqrt(np.nanmean((combined - y_true) ** 2)))
    results["base_rmse"] = base_rmse
    results["combined_rmse"] = combined_rmse
    results["rmse_improvement"] = base_rmse - combined_rmse

    return results
