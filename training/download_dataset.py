import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


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
    return cdsapi, mo, xr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download Data
    To ensure I don't blow up the limited memory on my device I will be downloading 1 month, cleaning, downsampling, saving and clearing old data until I have all 12 months stored in a .zarr file. Each day will be split into a few chunks to improve readabillity speed for training and compression to help reduce storage space.

    Region to be extracted:
    - North: 49.5°
    - West: -125°
    - South: 24.5°
    - East: -66.5°
    """)
    return


@app.cell
def _(cdsapi):
    dataset = "reanalysis-era5-land"
    request = {
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation"
        ],
        "year": "2025",
        "month": "01",
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [49.5, -125, 24.5, -66.5]
    }

    client = cdsapi.Client()
    return client, dataset, request


@app.cell
def _(client, dataset, request):
    client.retrieve(dataset, request).download()
    return


@app.cell
def _(xr):
    ds = xr.open_dataset("./85de23c247ffd5c702ce94477e713ba2.grib", engine="cfgrib")
    ds
    return (ds,)


@app.cell
def _(ds):
    ds.isel(time = 0, step = 23)["t2m"].plot()
    return


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
    ds_clean["t2m"].mean(dim=["latitude", "longitude"]).plot(marker="o")
    return


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
def _(mo):
    mo.md(r"""
    Now that I have verified my data collection works, lets append the rest of the months to our .zarr file
    """)
    return


@app.cell
def _(client, dataset):
    # loop through each month except jan
    grib_files = []
    for i in range(1, 12):
        # month numbers are 0 indexed
        length = 30
        if i == 1:
            length = 28 # febuary
        elif i in [2, 4, 6, 7, 9, 11]: 
            length = 31 # any month with 31 days [Mar, May, ...]

        days = [f"{i+1:02d}" for i in range(length)]

        new_request = {
        "variable": [
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation"
        ],
        "year": "2025",
        "month": f"{i+1:02d}",
        "day": days,
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": [49.5, -125, 24.5, -66.5]
        }

        print(f"Requesting month {i+1}")
        grib_files.append(client.retrieve(dataset, new_request).download())
        print(f"Recived month {i+1}: {grib_files[-1]}")
    print(grib_files)
    return (grib_files,)


@app.cell
def _(grib_files, xr):
    for file in grib_files: 
        file = "./" + file
        ds_new = xr.open_dataset("./85de23c247ffd5c702ce94477e713ba2.grib", engine="cfgrib")
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


    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
