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
    import torch
    from torch import nn, utils, optim
    from src.models.convLSTM_parts import ConvLSTM
    from src.engine.training import train_model, sucessive_halving

    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.data.read_data import WeatherTrainingData, create_transform_function
    from src.models.convLSTM import baseline_ConvLSTM
    from src.engine.eval_model import evaluate_model
    return (
        WeatherTrainingData,
        baseline_ConvLSTM,
        create_transform_function,
        evaluate_model,
        nn,
        optim,
        sucessive_halving,
        torch,
        train_model,
        utils,
    )


@app.cell
def _(WeatherTrainingData, create_transform_function, utils):
    var_transforms = {
        "sp": ["0to1"], 
        "t2m": ["z-scale"], 
        "tcc": [], 
        "tp": ["m-to-mm", "logp1"], 
        "u10": ["-1to1"], 
        "v10": ["-1to1"]
    }
    transforms = create_transform_function("./data/era5_conus_2025_1_to_2025_12_6var.zarr/",transforms=var_transforms)

    training_data = WeatherTrainingData(dir="./data/era5_conus_2025_1_to_2025_12_6var.zarr", set_type="train", transform=transforms)
    val_data = WeatherTrainingData(dir="./data/era5_conus_2025_1_to_2025_12_6var.zarr", set_type="validation", transform=transforms)
    training_dataloader = utils.data.DataLoader(training_data, batch_size=128, shuffle=True)
    val_dataloader = utils.data.DataLoader(val_data, batch_size=128, shuffle=True)
    return training_dataloader, val_dataloader


@app.cell
def _(nn, optim):
    params = {
        "model_input_dims": [6],
        "model_hidden_dim": [12, 16],
        "model_kernel_size": [3, 5],
        "model_num_layers": [1, 2],
        "model_num_pred_steps": [3],
        "optimizer": [optim.AdamW],
        "optim_lr": [0.0001],
        "loss": [nn.MSELoss]
    }
    return (params,)


@app.cell
def _(
    baseline_ConvLSTM,
    params,
    sucessive_halving,
    training_dataloader,
    val_dataloader,
):
    sucessive_halving(
        model_class=baseline_ConvLSTM,
        training_dataset=training_dataloader,
        validation_dataset=val_dataloader,
        epochs=2,
        hyper_parameters=params
    )
    return


@app.cell
def _():
    #weight_path = "./model_1.pth"
    #model_test = baseline_ConvLSTM(6, 12, 3, 2, 3)
    #checkpoint2 = torch.load(weight_path)
    #model_test.load_state_dict(checkpoint2['model_state_dict'])
    #model_test = torch.compile(model_test)
    #print(evaluate_model(model=model_test, dataset=val_dataloader))
    return


@app.cell
def _(baseline_ConvLSTM, evaluate_model, torch, val_dataloader):
    model_weights = [f"./model_{i}.pth" for i in range(8)]
    t_lay = [1,2,1,2,1,2,1,2]
    k_size = [3,3,5,5,3,3,5,5]
    h_dim = [12,12,12,12,16,16,16,16]
    for weights, t, k, h in zip(model_weights,t_lay,k_size,h_dim):
        model = baseline_ConvLSTM(6, h, k, t, 3)
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = torch.compile(model)
        print(evaluate_model(model=model, dataset=val_dataloader))
    return


@app.cell
def _(
    baseline_ConvLSTM,
    nn,
    optim,
    torch,
    train_model,
    training_dataloader,
    val_dataloader,
):
    # quick test
    model_test = baseline_ConvLSTM(
        input_dims=6,
        hidden_dim=12,
        kernel_size=3,
        num_layers=2,
        num_pred_steps=3
    )
    model_test = torch.compile(model_test)
    test_results = train_model(
        model=model_test,
        training_dataset=training_dataloader,
        validation_dataset=val_dataloader,
        epochs=30,
        loss_fun=nn.MSELoss(),
        optimizer=optim.AdamW(model_test.parameters(), lr=0.0001)
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
