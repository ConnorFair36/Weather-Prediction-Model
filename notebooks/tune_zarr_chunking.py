import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import xarray as xr
    import zarr
    import matplotlib.pyplot as plt
    import seaborn as sns
    return (xr,)


@app.cell
def _(xr):
    z_file = xr.open_dataset("./data/era5_conus_2025_1_to_2025_12_6var.zarr/")
    z_file
    return (z_file,)


@app.cell
def _(z_file):
    print(f"{z_file.isel(time=slice(0,4)).nbytes:,d}")
    return


@app.cell
def _(z_file):
    z_file["time"].size
    return


@app.cell
def _(z_file):
    z_file.dims
    return


if __name__ == "__main__":
    app.run()
