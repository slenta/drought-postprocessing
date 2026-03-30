from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def _month_sequence(start_month: int, end_month: int) -> list[int]:
    if not (1 <= int(start_month) <= 12 and 1 <= int(end_month) <= 12):
        raise ValueError("start_month and end_month must be in [1, 12].")

    months = [int(start_month)]
    while months[-1] != int(end_month):
        next_month = (months[-1] % 12) + 1
        months.append(next_month)

    return months


def combine_cwb_leadmonths_by_month_range(
    leadmonth_to_file: dict[int, str | Path],
    start_month: int,
    end_month: int,
    out_path: str | Path,
) -> Path:
    target_months = _month_sequence(start_month=start_month, end_month=end_month)

    month_chunks = []
    for month_offset, target_month in enumerate(target_months):
        source_leadmonth = int(month_offset) + 1
        source_path = Path(leadmonth_to_file[source_leadmonth])
        with xr.open_dataset(source_path) as source_ds:
            time_values = pd.to_datetime(source_ds["time"].values)
            month_mask = np.asarray(time_values.month == int(target_month))
            selected_ds = source_ds.isel(time=month_mask).load()
            selected_time = pd.to_datetime(selected_ds["time"].values)
            fitted_time = pd.DatetimeIndex(
                [pd.Timestamp(t.year, int(target_month), 1) for t in selected_time]
            )
            month_chunks.append(selected_ds.assign_coords(time=fitted_time))

    merged = xr.concat(month_chunks, dim="time").sortby("time")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(
        out_path,
        encoding={
            "time": {
                "units": "days since 1900-01-01",
                "calendar": "standard",
                "dtype": "f8",
            }
        },
    )
    merged.close()
    return out_path
