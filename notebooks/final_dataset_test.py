import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Examine the New Dataset

    This notebook is being used to determine the structure of the new dataset (reanalysis era5 single-levels)
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import dask
    import zarr
    import gribapi
    return mo, xr


@app.cell
def _():
    (731*12) - 12
    return


@app.cell
def _():
    365*2
    return


@app.cell
def _(xr):
    ds1 = xr.open_dataset("../data/643ed60b8171b91d0545f4c48256cac9.grib",engine="cfgrib",backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
    ds1
    return (ds1,)


@app.cell
def _(ds1):
    ds1["step"]
    return


@app.cell
def _(ds1):
    ds1.isel(time=730,step=4)["tp"].plot()
    return


@app.cell
def _(ds1):
    ds_stacked = ds1.assign_coords(
            valid_time=ds1["time"] + ds1["step"]
    )
    # compress time & step into 1 dimention
    ds_stacked = ds_stacked.stack(t=("time", "step"))
    # remove all entries where all values are nan
    ds_stacked = ds_stacked.dropna(dim="t", how="all")
    # remove unessicary columns?
    ds_clean = (
        ds_stacked
        .set_index(t="valid_time")
        .rename({"t": "valid_time"})
        .drop_vars(["time", "step"])
        .sortby("valid_time")
        .rename({"valid_time": "time"})
    )
    ds_clean
    return (ds_clean,)


@app.cell
def _(xr):
    ds2 = xr.open_dataset("../data/643ed60b8171b91d0545f4c48256cac9.grib",engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': ['10u', '10v', '2t', 'sp', 'tcc']}})
    ds2
    return (ds2,)


@app.cell
def _():
    21*4
    return


@app.cell
def _(ds2, ds_clean, xr):
    full_ds = xr.merge([ds_clean, ds2])
    full_ds
    return (full_ds,)


@app.cell
def _(full_ds):
    full_ds.isel(time=0)["t2m"].plot()
    return


@app.cell
def _(full_ds):
    print(full_ds.nbytes)
    print(full_ds.chunks)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
