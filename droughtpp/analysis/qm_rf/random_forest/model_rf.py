import typing as _t
import numpy as np
import xarray as xr
import pandas as pd
import time
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


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
        self.model: _t.Optional[_t.Union[RandomForestRegressor, XGBRegressor]] = None
        self.training_history: _t.Optional[dict] = None

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
        n_estimators: int = 100,
        random_state: int = 0,
    ) -> RandomForestRegressor:
        """Train and store a RandomForestRegressor."""
        rf = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1
        )
        rf.fit(X.values, y.values)
        self.model = rf
        return rf

    def train_with_eta(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: _t.Optional[pd.DataFrame] = None,
        y_val: _t.Optional[pd.Series] = None,
        model_type: str = "rf",
        n_estimators: int = 100,
        chunk_size: int = 10,
        random_state: int = 0,
        xgb_learning_rate: float = 0.1,
        xgb_max_depth: int = 6,
        xgb_subsample: float = 0.8,
        xgb_colsample_bytree: float = 0.8,
        xgb_min_child_weight: float = 1.0,
        xgb_eval_metric: str = "rmse",
    ) -> _t.Union[RandomForestRegressor, XGBRegressor]:

        model_type = str(model_type).lower()
        if model_type in {"xgb", "xgboost"}:
            xgb = XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                learning_rate=xgb_learning_rate,
                max_depth=xgb_max_depth,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                min_child_weight=xgb_min_child_weight,
                eval_metric=xgb_eval_metric,
                objective="reg:squarederror",
                n_jobs=-1,
            )
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
            self.model = xgb
            return xgb

        start = time.time()
        total_done = 0
        first = min(chunk_size, n_estimators)

        rf = RandomForestRegressor(
            n_estimators=first,
            warm_start=True,
            n_jobs=-1,
            random_state=random_state,
        )
        rf.fit(X.values, y.values)
        total_done += first

        train_rmse = [float(np.sqrt(np.mean((rf.predict(X.values) - y.values) ** 2)))]
        val_rmse = []
        if X_val is not None and y_val is not None:
            val_rmse.append(
                float(np.sqrt(np.mean((rf.predict(X_val.values) - y_val.values) ** 2)))
            )
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
            train_rmse.append(
                float(np.sqrt(np.mean((rf.predict(X.values) - y.values) ** 2)))
            )
            if X_val is not None and y_val is not None:
                val_rmse.append(
                    float(
                        np.sqrt(np.mean((rf.predict(X_val.values) - y_val.values) ** 2))
                    )
                )

        pbar.close()
        self.training_history = {
            "model_type": "rf",
            "metric": "rmse",
            "x": trees_hist,
            "train": train_rmse,
            "val": val_rmse if len(val_rmse) > 0 else None,
        }
        self.model = rf
        return rf

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


# ...existing code...
