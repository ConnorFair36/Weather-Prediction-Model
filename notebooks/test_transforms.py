import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    # Walk up one level from notebooks/ to reach the project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import matplotlib.pyplot as plt
    import seaborn as sns
    import xarray as xr
    import zarr

    import torch
    from torch import utils, nn
    import torch.nn.functional as F

    from src.data.read_data import WeatherTrainingData, generate_transforms, create_transform_function
    return (
        WeatherTrainingData,
        create_transform_function,
        generate_transforms,
        plt,
        torch,
        utils,
        xr,
    )


@app.cell
def _(generate_transforms):
    generate_transforms("./data/era5_conus_2025_1_to_2025_12_6var.zarr/", 
                        lda=["sp"],  # surface pressure needs lambda for boxcox transformation
                        mean=["t2m"], std=["t2m"], # temperature uses z-scalar transformation
                        min=["u10", "v10"], max=["u10", "v10"]) # wind u and v components use [-1, 1] transformation
    return


@app.cell
def _(xr):
    z_file = xr.open_dataset("./data/era5_conus_2025_1_to_2025_12_6var.zarr/")
    z_file
    return


@app.cell
def _(WeatherTrainingData, utils):
    training_data = WeatherTrainingData("./data/era5_conus_2025_1_to_2025_12_6var.zarr/", "train",seq_length=12)
    training_dataloader = utils.data.DataLoader(training_data, batch_size=128, num_workers=2)

    test_data = iter(training_dataloader)
    example_data = next(test_data)
    return (example_data,)


@app.cell
def _(example_data):
    # 2: longitude, 3: var, 0, time, 1: latitude
    example_data[1].shape
    return


@app.cell
def _(torch):
    torch.tensor([-4., -1., 0., 1., 2., 3.]) / max(abs(torch.tensor([-4., -1., 0., 1., 2., 3.]).min()), torch.tensor([-4., -1., 0., 1., 2., 3.]).max())
    return


@app.cell
def _(create_transform_function):
    var_transforms = {
        "sp": ["0to1"], 
        "t2m": ["z-scale"], 
        "tcc": [], 
        "tp": ["m-to-mm", "logp1"], 
        "u10": ["-1to1"], 
        "v10": ["-1to1"]
    }
    transforms = create_transform_function("./data/era5_conus_2025_1_to_2025_12_6var.zarr/",transforms=var_transforms)
    return (transforms,)


@app.cell
def _(transforms):
    type(transforms)
    return


@app.cell
def _(example_data, plt):
    plt.imshow(example_data[0][0,0,:,:,0].permute((1,0)))
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(example_data, transforms):
    t_data = transforms(example_data[0][0,:,:,:,:].clone())
    return (t_data,)


@app.cell
def _(plt, t_data):
    plt.imshow(t_data[0,:,:,0:1].permute((1,0,2)))
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(t_data):
    t_data[0,:,:,0:1].max()
    return


if __name__ == "__main__":
    app.run()
