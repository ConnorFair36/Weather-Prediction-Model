import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Preprocessing
    The purpose of this file is to take the downloaded .grib files and combile them into a single zarr dataset for faster access and to take up less memory on my device.

    This is only made to run if you have .grib files waiting to be processed.

    DO NOT RUN ON THE SAME DATASET MORE THAN ONCE
    """)
    return


@app.cell
def _():
    import marimo as mo
    import cdsapi
    import xarray as xr
    import numpy as np
    import gribapi
    import matplotlib.pyplot as plt
    import pandas as pd
    import dask
    import zarr
    return mo, xr


@app.cell
def _(xr):
    ds = xr.open_dataset("./68ec19223a9dab7b6140cfd3b0c30c04.grib", engine="cfgrib")
    ds
    return (ds,)


@app.cell
def _(ds):
    # clip NAN values off boty ends of the time series block
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
    ds_clean
    return (ds_clean,)


@app.cell
def _(ds_clean):
    # downsample all spatial data 4x
    ds_sampled = ds_clean.coarsen(latitude=4, boundary='pad').mean().coarsen(longitude=4, boundary='pad').mean()
    return (ds_sampled,)


@app.cell
def _(ds_sampled):
    ds_chunked = ds_sampled.chunk({
        "valid_time": 24,
        "latitude": 21,
        "longitude": 49
    })
    ds_chunked.chunks
    return (ds_chunked,)


@app.cell
def _(ds_chunked):
    ds_chunked.to_zarr(
        "./training/data/era5_conus_downsampled.zarr",
        mode="w",
        consolidated=True
    )
    return


@app.cell
def _(xr):
    grib_files = ["./a002da964f99ca46e1a73923e0f1ab8.grib",
                 "./25d0055e875c33c0036d6d1db3744884.grib",
                 "./df4ba52d02544db98cd7245c26735638.grib",
                 "./3fb4e5bc843b62e0da0709ec98b0342f.grib",
                 "./903b63464105c3fffe500b75dd931ffc.grib",
                 "./8af9baf536060d034c878b10cd388532.grib",
                 "./b0141f2b26204d561e23fa62d068809c.grib",
                 "./1508e9a54b3011f83468b98382740055.grib",
                 "./53a5b59988e716b9d2ea442f0a2529dd.grib",
                 "./8e9026b05b87cc64da301bdba82bfc8e.grib",
                 "./9af95a7fa353f3a0ac971cbeecbecf52.grib"]
    sanity_counter = 2
    for file in grib_files: 
        ds_new = xr.open_dataset(file, engine="cfgrib")
        # clip NAN values off boty ends of the time series block
        # create valid time
        ds_new_stacked = ds_new.assign_coords(
            valid_time=ds_new["time"] + ds_new["step"]
        )
        # compress time & step into 1 dimention
        ds_new_stacked = ds_new_stacked.stack(t=("time", "step"))
        # remove all entries where all values are nan
        ds_new_stacked = ds_new_stacked.dropna(dim="t", how="all")
        # remove unessicary columns?
        ds_new_clean = (
            ds_new_stacked
            .set_index(t="valid_time")
            .rename({"t": "valid_time"})
            .drop_vars(["time", "step"])
            .sortby("valid_time")
        )
        # downsample dataset 4x to make processing much faster
        ds_new_sampled = ds_new_clean.coarsen(latitude=4, boundary='pad').mean().coarsen(longitude=4, boundary='pad').mean()
        # divide data into smaller chunks for more efficent training
        ds_new_chunked = ds_new_sampled.chunk({
        "valid_time": 24,
        "latitude": 21,
        "longitude": 49
        })
        # append the new data o the existing .zarr file
        ds_new_chunked.to_zarr(
        "./training/data/era5_conus_downsampled.zarr",
        mode="a",
        append_dim="valid_time"
        )
        print(f"{sanity_counter} is done!")
        sanity_counter += 1
    return


@app.cell
def _(xr):
    full_data_test = xr.load_dataset("./training/data/era5_conus_downsampled.zarr",engine="zarr")
    full_data_test
    return


if __name__ == "__main__":
    app.run()
