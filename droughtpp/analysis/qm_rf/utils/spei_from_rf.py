"""
Compute SPEI from RF-corrected hindcasts by combining quantile-mapped data
with RF residuals, then calculating SPEI from the corrected CWB.

This module reuses the SPEI computation logic from spei_calc.py but applies
it to RF-postprocessed hindcasts.
"""

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import xarray as xr
import scipy.stats as sps
from droughtpp.evaluation.spei_calculation.spei_calc import compute_spei_from_surplus


def combine_qm_and_residuals(
    qm_data: xr.DataArray, residuals: xr.DataArray, var_name: str = "CWB"
) -> xr.DataArray:
    """
    Combine quantile-mapped hindcasts with RF residuals.

    Parameters
    ----------
    qm_data : xr.DataArray or xr.Dataset
        Quantile-mapped hindcast CWB data.
    residuals : xr.DataArray or xr.Dataset
        RF residuals (reference - qm).
    var_name : str
        Variable name to extract from datasets if needed.

    Returns
    -------
    xr.DataArray
        Combined data: qm_data + residuals
    """

    # Extract DataArrays if needed
    if isinstance(qm_data, xr.Dataset):
        qm_data = qm_data[var_name]
    if isinstance(residuals, xr.Dataset):
        residuals = residuals[var_name]

    # Align on common time/space coordinates
    qm_aligned, res_aligned = xr.align(qm_data, residuals, join="inner")

    # Combine
    combined = qm_aligned + res_aligned

    return combined


def compute_spei_from_rf_corrected(
    corrected_cwb: xr.DataArray,
    month_range: tuple = (1, 3),
    var_names: list = ["CWB", "spei"],
    dist=sps.fisk,
) -> xr.DataArray:
    """
    Compute SPEI from RF-corrected hindcast CWB (qm + residuals).

    Wraps compute_spei_from_surplus to work with RF-postprocessed data.

    Parameters
    ----------
    corrected_cwb : xr.DataArray
        Corrected CWB data (quantile-mapped + RF residuals) [mm].
        Must have dims ('time', 'latitude', 'longitude').
    month_range : tuple, optional
        1-based inclusive index range of months within each year over which
        to sum the surplus before computing SPEI. Example: (1,3) sums the
        first three months in each year.
    var_names : list, optional
        [input_var_name, output_var_name]. Default = ['CWB', 'spei'].
    dist : scipy.stats distribution, optional
        Distribution for SPEI standardization. Default = sps.fisk (log-logistic).

    Returns
    -------
    xr.DataArray
        Global SPEI with shape (time, latitude, longitude),
        standardized (mean=0, std=1).
    """

    # Use the existing compute_spei_from_surplus function
    spei = compute_spei_from_surplus(
        corrected_cwb, month_range=month_range, var_names=var_names, dist=dist
    )

    # Update attributes to indicate RF correction
    spei.attrs.update(
        {
            "processing": "quantile_mapping + RF_residuals",
            "description": f"SPEI computed from RF-corrected hindcasts (QM + residuals) over months {month_range}",
        }
    )

    return spei
