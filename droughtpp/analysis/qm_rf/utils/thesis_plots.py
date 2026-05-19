from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def _load_land_mask(land_mask_path: Path) -> xr.DataArray:
    with xr.open_dataset(land_mask_path) as ds:
        if "landmask" in ds.data_vars:
            mask = ds["landmask"]
        else:
            mask = next(iter(ds.data_vars.values()))

        while mask.ndim > 2:
            mask = mask.isel({mask.dims[0]: 0})

        return mask.load()


def _apply_land_mask(da: xr.DataArray, land_mask: xr.DataArray) -> xr.DataArray:
    spatial_dims = [
        d for d in da.dims if d not in {"time", "member", "ensemble", "ens"}
    ]
    mask = land_mask

    if len(mask.dims) == len(spatial_dims) and tuple(mask.dims) != tuple(spatial_dims):
        rename_map = {src: dst for src, dst in zip(mask.dims, spatial_dims)}
        mask = mask.rename(rename_map)

    return da.where(mask > 0)


def plot_excel_rows_vs_columns(
    excel_path: Path,
    output_path: Path,
    sheet_name: str | int = 0,
    ylabel: str = "Value",
    xlabel: str = "Columns",
    right_on: list | None = None,
    right_ylabel: str | None = None,
    figsize: tuple = (10, 6),
) -> Path:
    table = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise RuntimeError(
            "Excel table must have at least 2 rows and 2 columns (titles + data)."
        )

    title_cell = table.iloc[0, 0]
    title = "Excel Table Plot" if pd.isna(title_cell) else str(title_cell)

    col_titles = table.iloc[0, 1:].astype(str).tolist()
    row_titles = table.iloc[1:, 0].astype(str).tolist()
    values = table.iloc[1:, 1:].apply(pd.to_numeric, errors="coerce")

    x = np.arange(len(row_titles))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Partition columns (x-values) by axis: right_on specifies which column indices use the right axis.
    # Plot each row as one continuous line on the left axis, then overlay right-column markers on right axis.
    x = np.arange(len(col_titles))

    right_col_indices: set[int] = set()
    left_col_indices: set[int] = set()
    if right_on is not None:
        for idx in right_on:
            if isinstance(idx, int) and 0 <= idx < len(col_titles):
                right_col_indices.add(idx)
            else:
                left_col_indices.add(idx)

    ax2 = ax.twinx() if right_col_indices else None

    # Determine axis ranges so we can map right-axis data into left-axis scale for plotting a single continuous line
    all_left_vals = values.iloc[:, sorted(list(set(range(len(col_titles))) - right_col_indices))].to_numpy(dtype=float) if (set(range(len(col_titles))) - right_col_indices) else np.array([])
    all_right_vals = values.iloc[:, sorted(list(right_col_indices))].to_numpy(dtype=float) if right_col_indices else np.array([])

    # compute min/max while handling empty cases
    if all_left_vals.size:
        left_min = float(0.9 * np.nanmin(all_left_vals))
        left_max = float(1.1 * np.nanmax(all_left_vals))
    else:
        # fallback to overall data
        left_min = float(0.8 * np.nanmin(values.to_numpy()))
        left_max = float(1.2 * np.nanmax(values.to_numpy()))

    if all_right_vals.size:
        right_min = float(0.8 * np.nanmin(all_right_vals) - 0.1)
        right_max = float(1.2 * np.nanmax(all_right_vals))
    else:
        right_min = float(0.9 * left_min)
        right_max = float(1.1 * left_max)

    # set axis limits so ticks align between axes
    ax.set_ylim(left_min, left_max)
    if ax2 is not None:
        ax2.set_ylim(right_min, right_max)

    # Plot each row once: map right-column y values into left-axis scale so the line is continuous
    for row_idx, row_name in enumerate(row_titles):
        y = values.iloc[row_idx, :].to_numpy(dtype=float)
        y_plot = np.full_like(y, np.nan, dtype=float)

        # left columns: use original values
        left_cols = sorted(list(set(range(len(col_titles))) - right_col_indices))
        if left_cols:
            y_plot[left_cols] = y[left_cols]

        # right columns: map into left axis scale
        if right_col_indices:
            right_cols = sorted(list(right_col_indices))
            # linear mapping from right range to left range
            y_right = y[right_cols]
            y_mapped = left_min + (y_right - right_min) / (right_max - right_min) * (left_max - left_min)
            y_plot[right_cols] = y_mapped

        # plot continuous line (will connect across left/right columns using mapped values)
        h, = ax.plot(x, y_plot, marker="o", linewidth=1.8, label=row_name)

    ax.set_xticks(x)
    ax.set_xticklabels(col_titles, rotation=45, ha="right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if right_ylabel and ax2 is not None:
        ax2.set_ylabel(right_ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)

    # Legend from left axis (primary)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


class QMTimelinePlotter:
    def __init__(
        self,
        paths_root: Path,
        quantiles: list[int],
        leadmonth: int,
        var_name: str,
        output_path: Path,
        reference_data: Path,
        land_mask_path: Path,
        hindcasts_root: Path | None = None,
    ) -> None:
        self.paths_root = paths_root
        self.hindcasts_root = (
            hindcasts_root or (paths_root.parent / "lm{}".format(leadmonth)).parent
        )
        self.quantiles = quantiles
        self.leadmonth = leadmonth
        self.var_name = var_name
        self.output_path = output_path
        self.reference_data = reference_data
        self.land_mask_path = land_mask_path
        self.land_mask = _load_land_mask(land_mask_path)

    def _spatial_dims(self, da: xr.DataArray) -> list[str]:
        return [d for d in da.dims if d not in {"time", "member", "ensemble", "ens"}]

    def _load_qm_member_paths(self, n_quantiles: int) -> list[str]:
        json_path = (
            self.paths_root
            / f"nq{n_quantiles}"
            / f"lm{self.leadmonth}"
            / "qm_hindcast_paths.json"
        )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_original_hindcast_paths(self) -> list[str]:
        json_path = (
            self.hindcasts_root
            / f"lm{self.leadmonth}"
            / f"original_hindcast_lm{self.leadmonth}_paths.json"
        )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _timeline_for_quantile(self, n_quantiles: int) -> xr.DataArray:
        member_paths = self._load_qm_member_paths(n_quantiles)
        member_series: list[xr.DataArray] = []
        for p in member_paths:
            with xr.open_dataset(p) as ds:
                da = _apply_land_mask(ds[self.var_name], self.land_mask)
                spatial_dims = self._spatial_dims(da)
                s = da.mean(dim=spatial_dims, skipna=True).sortby("time")
                member_series.append(s)

        common = member_series[0]["time"].values
        for s in member_series[1:]:
            common = np.intersect1d(common, s["time"].values)

        trimmed = [s.sel(time=common) for s in member_series]
        members_da = xr.concat(trimmed, dim="member")
        return members_da

    def _timeline_for_reference(self) -> xr.DataArray:
        with xr.open_dataset(self.reference_data) as ds:
            da = _apply_land_mask(ds[self.var_name], self.land_mask)
            spatial_dims = self._spatial_dims(da)
            return da.mean(dim=spatial_dims, skipna=True).sortby("time")

    def _timeline_for_original(self) -> xr.DataArray:
        member_paths = self._load_original_hindcast_paths()
        member_series: list[xr.DataArray] = []
        for p in member_paths:
            with xr.open_dataset(p) as ds:
                da = _apply_land_mask(ds[self.var_name], self.land_mask)
                spatial_dims = self._spatial_dims(da)
                s = da.mean(dim=spatial_dims, skipna=True).sortby("time")
                member_series.append(s)

        common = member_series[0]["time"].values
        for s in member_series[1:]:
            common = np.intersect1d(common, s["time"].values)

        trimmed = [s.sel(time=common) for s in member_series]
        members_da = xr.concat(trimmed, dim="member")
        return members_da

    def run(self) -> Path:
        # timelines now hold per-member series (time x member)
        timelines: dict[int, xr.DataArray] = {
            n: self._timeline_for_quantile(n) for n in self.quantiles
        }
        ref_timeline = self._timeline_for_reference()
        orig_members = self._timeline_for_original()

        common_time = None
        for members_da in timelines.values():
            times = members_da["time"].values
            common_time = (
                times if common_time is None else np.intersect1d(common_time, times)
            )
        common_time = np.intersect1d(common_time, ref_timeline["time"].values)
        common_time = np.intersect1d(common_time, orig_members["time"].values)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 5))

        # Original hindcast: compute ensemble mean and spread
        orig_da = orig_members.sel(time=common_time)
        orig_mean = orig_da.mean(dim="member", skipna=True)
        orig_min = orig_da.min(dim="member", skipna=True)
        orig_max = orig_da.max(dim="member", skipna=True)

        plt.plot(
            orig_mean["time"].values,
            orig_mean.values,
            linewidth=2,
            linestyle="--",
            color="gray",
            label="Original",
        )
        plt.fill_between(
            orig_mean["time"].values,
            orig_min.values,
            orig_max.values,
            color="gray",
            alpha=0.2,
        )

        # QM quantiles: plot mean and min/max spread
        for n in self.quantiles:
            members_da = timelines[n].sel(time=common_time)
            mean_da = members_da.mean(dim="member", skipna=True)
            min_da = members_da.min(dim="member", skipna=True)
            max_da = members_da.max(dim="member", skipna=True)

            plt.plot(
                mean_da["time"].values, mean_da.values, linewidth=2, label=f"QM n={n}"
            )
            plt.fill_between(
                mean_da["time"].values,
                min_da.values,
                max_da.values,
                alpha=0.18,
            )

        ts_ref = ref_timeline.sel(time=common_time)
        plt.plot(
            ts_ref["time"].values,
            ts_ref.values,
            color="black",
            linewidth=2.5,
            label="Reference",
        )

        plt.xlabel("Time")
        plt.ylabel(self.var_name)
        plt.title(f"QM Timeline Comparison (lm{self.leadmonth})")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150)
        plt.close()

        return self.output_path


class QMDistributionPlotter:
    def __init__(
        self,
        paths_root: Path,
        quantiles: list[int],
        leadmonths: list[int],
        var_name: str,
        output_path: Path,
        reference_data: Path,
        land_mask_path: Path,
        bins: int = 250,
        hindcasts_root: Path | None = None,
    ) -> None:
        self.paths_root = paths_root
        self.hindcasts_root = hindcasts_root or paths_root.parent
        self.quantiles = quantiles
        self.leadmonths = leadmonths
        self.var_name = var_name
        self.output_path = output_path
        self.reference_data = reference_data
        self.land_mask_path = land_mask_path
        self.land_mask = _load_land_mask(land_mask_path)
        self.bins = bins

    def _load_qm_member_paths(self, n_quantiles: int, leadmonth: int) -> list[str]:
        json_path = (
            self.paths_root
            / f"nq{n_quantiles}"
            / f"lm{leadmonth}"
            / "qm_hindcast_paths.json"
        )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_original_hindcast_paths(self, leadmonth: int) -> list[str]:
        json_path = (
            self.hindcasts_root
            / f"lm{leadmonth}"
            / f"original_hindcast_lm{leadmonth}_paths.json"
        )
        with open(json_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _data_for_quantile(self, n_quantiles: int) -> xr.DataArray:
        leadmonth_data = []
        for leadmonth in self.leadmonths:
            member_paths = self._load_qm_member_paths(n_quantiles, leadmonth)
            ds = xr.open_mfdataset(member_paths, combine="nested", concat_dim="member")
            da = (
                _apply_land_mask(ds[self.var_name], self.land_mask)
                .load()
                .sortby("time")
            )
            ds.close()
            leadmonth_data.append(da)

        if len(leadmonth_data) == 1:
            return leadmonth_data[0]
        return (
            xr.concat(leadmonth_data, dim="time")
            .sortby("time")
            .drop_duplicates(dim="time")
        )

    def _data_for_original(self) -> xr.DataArray:
        leadmonth_data = []
        for leadmonth in self.leadmonths:
            member_paths = self._load_original_hindcast_paths(leadmonth)
            ds = xr.open_mfdataset(member_paths, combine="nested", concat_dim="member")
            da = (
                _apply_land_mask(ds[self.var_name], self.land_mask)
                .load()
                .sortby("time")
            )
            ds.close()
            leadmonth_data.append(da)

        if len(leadmonth_data) == 1:
            return leadmonth_data[0]
        return (
            xr.concat(leadmonth_data, dim="time")
            .sortby("time")
            .drop_duplicates(dim="time")
        )

    def _get_reference_data(self) -> xr.DataArray:
        with xr.open_dataset(self.reference_data) as ds:
            return (
                _apply_land_mask(ds[self.var_name], self.land_mask)
                .load()
                .sortby("time")
            )

    def run(self) -> Path:
        data_by_n: dict[int, xr.DataArray] = {
            n: self._data_for_quantile(n) for n in self.quantiles
        }
        ref_data = self._get_reference_data()
        orig_data = self._data_for_original()

        common_time = None
        for da in data_by_n.values():
            times = da["time"].values
            common_time = (
                times if common_time is None else np.intersect1d(common_time, times)
            )
        common_time = np.intersect1d(common_time, ref_data["time"].values)
        common_time = np.intersect1d(common_time, orig_data["time"].values)

        values_by_n: dict[int, np.ndarray] = {}
        for n, da in data_by_n.items():
            vals = da.sel(time=common_time).values.ravel()
            values_by_n[n] = vals[np.isfinite(vals)]

        vals = ref_data.sel(time=common_time).values.ravel()
        ref_vals = vals[np.isfinite(vals)]

        vals = orig_data.sel(time=common_time).values.ravel()
        orig_vals = vals[np.isfinite(vals)]

        all_arrays = [orig_vals] + list(values_by_n.values()) + [ref_vals]
        all_values = np.concatenate(all_arrays)
        x_min = float(np.nanmin(all_values))
        x_max = float(np.nanmax(all_values))
        bins_array = np.linspace(x_min, x_max, self.bins)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 6))

        plt.hist(
            orig_vals,
            bins=bins_array,
            density=True,
            alpha=0.35,
            linestyle="--",
            color="gray",
            label="Original",
        )

        for n in self.quantiles:
            plt.hist(
                values_by_n[n],
                bins=bins_array,
                density=True,
                alpha=0.35,
                label=f"QM n={n}",
            )

        plt.hist(
            ref_vals,
            bins=bins_array,
            density=True,
            alpha=0.35,
            color="black",
            label="Reference",
        )

        plt.xlabel(self.var_name)
        plt.ylabel("Density")
        plt.title("QM Distribution Comparison (all leadmonths combined)")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_path, dpi=150)
        plt.close()

        return self.output_path


def _parse_args() -> argparse.Namespace:
    default_paths_root = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/paths/CWB"
    )
    default_ref = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/data/preprocessing/reference_cwb/reference_cwb_000.nc"
    )
    default_output = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/plots/thesis/timeline_qm_nq_compare_lm1.png"
    )
    default_distribution_output = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/plots/thesis/distribution_qm_nq_compare_all_leadmonths.png"
    )
    default_land_mask = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/data/spei/landmask.nc"
    )

    parser = argparse.ArgumentParser(
        description="Plot thesis QM comparison figures for multiple n_quantiles."
    )
    parser.add_argument(
        "--paths-root",
        type=Path,
        default=default_paths_root,
        help="Root folder containing nq*/lm*/qm_hindcast_paths.json",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        nargs="+",
        default=[2, 3, 5, 15, 20, 100],
        help="List of n_quantiles to compare",
    )
    parser.add_argument("--leadmonth", type=int, default=1, help="Lead month to plot")
    parser.add_argument(
        "--leadmonths",
        type=int,
        nargs="+",
        default=[1],
        help="Lead months used for distribution mode",
    )
    parser.add_argument(
        "--var-name", type=str, default="CWB", help="Variable name in NetCDF files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output timeline figure path",
    )
    parser.add_argument(
        "--output-distribution",
        type=Path,
        default=default_distribution_output,
        help="Output distribution figure path",
    )
    parser.add_argument(
        "--bins", type=int, default=250, help="Histogram bins for distribution mode"
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=default_ref,
        help="Optional reference dataset path; set to empty string to disable",
    )
    parser.add_argument(
        "--land-mask-path",
        type=Path,
        default=default_land_mask,
        help="Path to land mask NetCDF applied before spatial averaging",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # plotter_tl = QMTimelinePlotter(
    #     paths_root=args.paths_root,
    #     quantiles=args.quantiles,
    #     leadmonth=args.leadmonth,
    #     var_name=args.var_name,
    #     output_path=args.output,
    #     reference_data=args.reference_data,
    #     land_mask_path=args.land_mask_path,
    # )
    # timeline_path = plotter_tl.run()
    # print(f"Timeline: {timeline_path}")

    # plotter_dist = QMDistributionPlotter(
    #     paths_root=args.paths_root,
    #     quantiles=args.quantiles,
    #     leadmonths=args.leadmonths,
    #     var_name=args.var_name,
    #     output_path=args.output_distribution,
    #     reference_data=args.reference_data,
    #     land_mask_path=args.land_mask_path,
    #     bins=args.bins,
    # )
    # distribution_path = plotter_dist.run()
    # print(f"Distribution: {distribution_path}")

    # Plot Excel skill scores table
    # excel_path = Path(
    #     "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/data/CWB/skill_scores.xlsx"
    # )
    # excel_out = Path(
    #     "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/plots/thesis/skill_scores_plot.png"
    # )
    excel_path = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/data/CWB/lm_ts.xlsx"
    )
    excel_out = Path(
        "/work/bk1318/k202208/crai/hindcast-pp/Ml-Drought-Postprocessing/droughtpp/eval_results/cwb/qm_rf/plots/thesis/lm_ts.png"
    )
    excel_out.parent.mkdir(parents=True, exist_ok=True)
    # plot_excel_rows_vs_columns(excel_path=excel_path, output_path=excel_out, right_on=[1, 2])
    plot_excel_rows_vs_columns(excel_path=excel_path, output_path=excel_out)
    print(f"Excel plot: {excel_out}")


if __name__ == "__main__":
    main()
