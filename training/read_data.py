import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import xarray as xr
    import numpy as np
    import gribapi
    import matplotlib.pyplot as plt
    import pandas as pd
    return mo, np, xr


@app.cell
def _(xr):
    ds = xr.open_dataset("./f984e1115079d9c8f213746f9fc45def.grib", engine="cfgrib")
    ds
    return (ds,)


@app.cell
def _(ds):
    print(ds.data_vars)
    print(ds.coords)
    return


@app.cell
def _(ds):
    ds["step"]
    return


@app.cell
def _(ds):
    ds.isel(time=0, step=23)
    return


@app.cell
def _(ds, np):
    ds_1h = ds.sel(step=np.timedelta64(1, "h"))
    ds_1h["t2m"].isel(time=1).plot()

    return


@app.cell
def _(ds):
    tp = ds["tp"]
    return (tp,)


@app.cell
def _(tp):
    print(tp.shape)
    print(tp.dims)
    return


@app.cell
def _(tp):
    tp.isel(time=1, step=0)
    return


@app.cell
def _(ds):
    # create valid time
    ds_stacked = ds.assign_coords(
        valid_time=ds["time"] + ds["step"]
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
    )
    # profit
    ds_clean.isel(valid_time=0)
    return (ds_clean,)


@app.cell
def _(ds_clean):
    ds_clean["t2m"].mean(dim=["latitude", "longitude"]).plot(marker="o")

    return


@app.cell
def _(ds_clean):
    ds_sampled = ds_clean.coarsen(latitude=4, boundary='pad').mean().coarsen(longitude=4, boundary='pad').mean()
    return (ds_sampled,)


@app.cell
def _(mo):
    slider = mo.ui.slider(start=0, stop=23, label="Slider", value=3)
    return (slider,)


@app.cell
def _(mo, slider):
    mo.hstack([slider, mo.md(f"Has value: {slider.value}")])
    return


@app.cell
def _(ds_sampled, slider):
    ds_sampled["tp"].isel(valid_time=slider.value).plot()
    return


@app.cell
def _(ds_sampled):
    ds_sampled
    return


if __name__ == "__main__":
    app.run()
