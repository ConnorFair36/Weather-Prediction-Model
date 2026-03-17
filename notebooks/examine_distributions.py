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
    from sklearn.preprocessing import PowerTransformer
    from scipy import stats 
    return np, stats, xr


@app.cell
def _(xr):
    z_file = xr.open_dataset("./data/era5_conus_2025_1_to_2025_12_6var.zarr/")
    z_file
    return (z_file,)


@app.cell
def _(z_file):
    z_file["t2m"].plot.hist()
    return


@app.cell
def _(z_file):
    m = z_file["t2m"].mean()
    std = z_file["t2m"].std()
    z_file["t2m"].pipe(lambda x: (x - m) / std).plot.hist()
    return


@app.cell
def _(z_file):
    z_file["sp"].plot.hist()
    return


@app.cell
def _(np, stats, z_file):
    sample = np.random.choice(z_file["sp"].values.flatten(), size=200_000, replace=False)
    _, l = stats.boxcox(sample)
    return (l,)


@app.cell
def _(l, z_file):
    z_file["sp"].pipe(lambda x: ((x ** l) - 1)/l).plot.hist()
    return


@app.cell
def _(z_file):
    max_sp = z_file["sp"].max()
    z_file["sp"].pipe(lambda x: (max_sp + 1 - x)**(9/15)).plot.hist()
    return


if __name__ == "__main__":
    app.run()
