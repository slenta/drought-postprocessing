import numpy as np
import pandas as pd


def compute_leadmonth_mask(time_values, leadmonth: int) -> np.ndarray:
    timestamps = pd.to_datetime(time_values)
    years = np.asarray(timestamps.year, dtype=int)
    months = np.asarray(timestamps.month, dtype=int)

    first_month_by_year = {}
    for year in np.unique(years):
        first_month_by_year[int(year)] = int(months[years == year].min())

    lead_months = np.array(
        [
            ((int(month) - first_month_by_year[int(year)]) % 12) + 1
            for year, month in zip(years, months)
        ],
        dtype=int,
    )

    return lead_months == int(leadmonth)
