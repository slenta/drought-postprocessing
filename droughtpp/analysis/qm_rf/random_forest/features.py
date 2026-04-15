from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class RFFeatureBuilder:
    def __init__(
        self,
        reference_data_path: Path,
        var_name: str,
        additional_reference_features=None,
        feature_flags: dict | None = None,
    ):
        with xr.open_dataset(reference_data_path) as ds:
            reference_data_array = ds[var_name].load()

        self.reference_data_array = self._ensure_spatial_dims(
            reference_data_array
        ).squeeze()
        self.reference_values = np.asarray(self.reference_data_array.values)

        reference_time = pd.to_datetime(self.reference_data_array["time"].values)
        self.reference_time_index = {
            (int(timestamp.year), int(timestamp.month)): index
            for index, timestamp in enumerate(reference_time)
        }

        reference_latitudes = np.asarray(self.reference_data_array["latitude"].values)
        reference_longitudes = np.asarray(self.reference_data_array["longitude"].values)

        self.reference_lat_index = {
            round(float(value), 6): index
            for index, value in enumerate(reference_latitudes)
        }
        self.reference_lon_index = {
            round(float(value), 6): index
            for index, value in enumerate(reference_longitudes)
        }
        # obs_prev lags now only via additional_reference_features
        raw_feature_flags = feature_flags or {}
        self.feature_flags = {
            "use_predictor": bool(raw_feature_flags.get("use_predictor", True)),
            "use_lat": bool(raw_feature_flags.get("use_lat", True)),
            "use_lon": bool(raw_feature_flags.get("use_lon", True)),
            "use_lead_month": bool(raw_feature_flags.get("use_lead_month", True)),
        }

        self.additional_reference_features = []
        additional_reference_features = additional_reference_features or []
        for feature_cfg in additional_reference_features:
            feature_name = str(feature_cfg["name"])
            feature_path = Path(feature_cfg["path"])
            feature_var_name = str(feature_cfg["var_name"])
            feature_mode = str(feature_cfg.get("mode", "lagged")).strip().lower()

            with xr.open_dataset(feature_path) as ds_feature:
                feature_data_array = ds_feature[feature_var_name].load()

            feature_data_array = self._ensure_spatial_dims(feature_data_array).squeeze()
            feature_latitudes = np.asarray(feature_data_array["latitude"].values)
            feature_longitudes = np.asarray(feature_data_array["longitude"].values)

            feature_lat_index = {
                round(float(value), 6): index
                for index, value in enumerate(feature_latitudes)
            }
            feature_lon_index = {
                round(float(value), 6): index
                for index, value in enumerate(feature_longitudes)
            }

            if feature_mode == "monthly_climatology":
                feature_monthly = feature_data_array.groupby("time.month").mean(
                    "time", skipna=True
                )
                feature_monthly_values = np.asarray(feature_monthly.values)

                self.additional_reference_features.append(
                    {
                        "name": feature_name,
                        "mode": feature_mode,
                        "monthly_values": feature_monthly_values,
                        "lat_index": feature_lat_index,
                        "lon_index": feature_lon_index,
                    }
                )
                continue

            feature_values = np.asarray(feature_data_array.values)
            feature_time = pd.to_datetime(feature_data_array["time"].values)
            feature_time_index = {
                (int(timestamp.year), int(timestamp.month)): index
                for index, timestamp in enumerate(feature_time)
            }

            self.additional_reference_features.append(
                {
                    "name": feature_name,
                    "mode": feature_mode,
                    "values": feature_values,
                    "time_index": feature_time_index,
                    "lat_index": feature_lat_index,
                    "lon_index": feature_lon_index,
                }
            )

    @staticmethod
    def _ensure_spatial_dims(data_array: xr.DataArray) -> xr.DataArray:
        rename_map = {}
        if "lat" in data_array.dims:
            rename_map["lat"] = "latitude"
        if "lon" in data_array.dims:
            rename_map["lon"] = "longitude"
        if rename_map:
            data_array = data_array.rename(rename_map)
        return data_array

    @staticmethod
    def _compute_lead_months(years: np.ndarray, months: np.ndarray):
        first_month_by_year = {}
        for year in np.unique(years):
            first_month_by_year[int(year)] = int(months[years == year].min())

        lead_months = np.array(
            [
                ((int(month) - first_month_by_year[int(year)]) % 12) + 1
                for year, month in zip(years, months)
            ],
            dtype=float,
        )

        return lead_months, first_month_by_year

    @staticmethod
    def _shift_year_month(year: int, month: int, lag: int):
        shifted_year = int(year)
        shifted_month = int(month) - int(lag)

        while shifted_month <= 0:
            shifted_month += 12
            shifted_year -= 1

        return shifted_year, shifted_month

    def _lookup_coordinate_indices(
        self, values: np.ndarray, index_map: dict
    ) -> np.ndarray:
        return np.array(
            [index_map.get(round(float(value), 6), -1) for value in values], dtype=int
        )

    def build(self, predictor_data_array: xr.DataArray):
        predictor_data_array = self._ensure_spatial_dims(predictor_data_array).squeeze()
        predictor_stacked = predictor_data_array.stack(
            sample=list(predictor_data_array.dims)
        )

        timestamps = pd.to_datetime(predictor_stacked.coords["time"].values)
        years = np.asarray(timestamps.year, dtype=int)
        months = np.asarray(timestamps.month, dtype=int)

        lead_months, first_month_by_year = self._compute_lead_months(years, months)

        latitudes = np.asarray(predictor_stacked.coords["latitude"].values)
        longitudes = np.asarray(predictor_stacked.coords["longitude"].values)

        lat_indices = self._lookup_coordinate_indices(
            latitudes, self.reference_lat_index
        )
        lon_indices = self._lookup_coordinate_indices(
            longitudes, self.reference_lon_index
        )

        obs_prev_1 = np.full(predictor_stacked.sizes["sample"], np.nan, dtype=float)
        obs_prev_2 = np.full(predictor_stacked.sizes["sample"], np.nan, dtype=float)
        obs_prev_3 = np.full(predictor_stacked.sizes["sample"], np.nan, dtype=float)

        valid_spatial = (lat_indices >= 0) & (lon_indices >= 0)



        additional_lag_features = {}
        for feature in self.additional_reference_features:
            feature_name = feature["name"]
            feature_mode = feature.get("mode", "lagged")
            feature_lat_index = feature["lat_index"]
            feature_lon_index = feature["lon_index"]

            feature_lat_indices = self._lookup_coordinate_indices(
                latitudes, feature_lat_index
            )
            feature_lon_indices = self._lookup_coordinate_indices(
                longitudes, feature_lon_index
            )
            feature_valid_spatial = (feature_lat_indices >= 0) & (
                feature_lon_indices >= 0
            )

            if feature_mode == "monthly_climatology":
                feature_monthly_values = feature["monthly_values"]
                feature_clim = np.full(
                    predictor_stacked.sizes["sample"], np.nan, dtype=float
                )
                sample_indices = np.where(feature_valid_spatial)[0]
                if sample_indices.size > 0:
                    sample_month_indices = months[sample_indices] - 1
                    sample_lat_indices = feature_lat_indices[sample_indices]
                    sample_lon_indices = feature_lon_indices[sample_indices]
                    feature_clim[sample_indices] = feature_monthly_values[
                        sample_month_indices,
                        sample_lat_indices,
                        sample_lon_indices,
                    ]
                additional_lag_features[f"{feature_name}_clim_month"] = feature_clim
                continue

            feature_values = feature["values"]
            feature_time_index = feature["time_index"]

            feature_prev_1 = np.full(
                predictor_stacked.sizes["sample"], np.nan, dtype=float
            )
            feature_prev_2 = np.full(
                predictor_stacked.sizes["sample"], np.nan, dtype=float
            )
            feature_prev_3 = np.full(
                predictor_stacked.sizes["sample"], np.nan, dtype=float
            )

            for year, first_month in first_month_by_year.items():
                year_mask = years == int(year)
                sample_mask = year_mask & feature_valid_spatial

                if not np.any(sample_mask):
                    continue

                sample_indices = np.where(sample_mask)[0]
                sample_lat_indices = feature_lat_indices[sample_indices]
                sample_lon_indices = feature_lon_indices[sample_indices]

                for lag, target_array in (
                    (1, feature_prev_1),
                    (2, feature_prev_2),
                    (3, feature_prev_3),
                ):
                    ref_year, ref_month = self._shift_year_month(year, first_month, lag)
                    time_index = feature_time_index.get((ref_year, ref_month))
                    if time_index is None:
                        continue

                    feature_slice = feature_values[time_index]
                    target_array[sample_indices] = feature_slice[
                        sample_lat_indices,
                        sample_lon_indices,
                    ]

            additional_lag_features[f"{feature_name}_prev_1"] = feature_prev_1
            additional_lag_features[f"{feature_name}_prev_2"] = feature_prev_2
            additional_lag_features[f"{feature_name}_prev_3"] = feature_prev_3

        n_samples = predictor_stacked.sizes["sample"]
        feature_columns = {}
        if self.feature_flags["use_predictor"]:
            feature_columns["predictor"] = predictor_stacked.values
        if self.feature_flags["use_lat"]:
            feature_columns["lat"] = latitudes
        if self.feature_flags["use_lon"]:
            feature_columns["lon"] = longitudes
        if self.feature_flags["use_lead_month"]:
            feature_columns["lead_month"] = lead_months



        features = pd.DataFrame(index=np.arange(n_samples))
        if feature_columns:
            features = pd.concat([features, pd.DataFrame(feature_columns)], axis=1)
        if additional_lag_features:
            features = pd.concat(
                [features, pd.DataFrame(additional_lag_features)], axis=1
            )

        if features.shape[1] == 0:
            raise ValueError("No ML features enabled. Check feature_flags in config.")

        return features, years
