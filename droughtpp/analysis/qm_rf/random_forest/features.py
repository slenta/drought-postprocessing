from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


class RFFeatureBuilder:
    def __init__(self, reference_data_path: Path, var_name: str):
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

        for year, first_month in first_month_by_year.items():
            year_mask = years == int(year)
            sample_mask = year_mask & valid_spatial

            if not np.any(sample_mask):
                continue

            sample_indices = np.where(sample_mask)[0]
            sample_lat_indices = lat_indices[sample_indices]
            sample_lon_indices = lon_indices[sample_indices]

            for lag, target_array in (
                (1, obs_prev_1),
                (2, obs_prev_2),
                (3, obs_prev_3),
            ):
                ref_year, ref_month = self._shift_year_month(year, first_month, lag)
                time_index = self.reference_time_index.get((ref_year, ref_month))
                if time_index is None:
                    continue

                reference_slice = self.reference_values[time_index]
                target_array[sample_indices] = reference_slice[
                    sample_lat_indices,
                    sample_lon_indices,
                ]

        features = pd.DataFrame(
            {
                "predictor": predictor_stacked.values,
                "lat": latitudes,
                "lon": longitudes,
                "lead_month": lead_months,
                "obs_prev_1": obs_prev_1,
                "obs_prev_2": obs_prev_2,
                "obs_prev_3": obs_prev_3,
            }
        )

        return features, years
