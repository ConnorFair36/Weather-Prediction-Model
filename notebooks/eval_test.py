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
    from src.engine.training import train_model

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
        np,
        optim,
        plt,
        torch,
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
    return (val_dataloader,)


@app.cell
def _(baseline_ConvLSTM, optim, torch):
    # 1. Initialize model and optimizer as usual
    model = baseline_ConvLSTM(6, 12, 3, 2, 3)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 2. Load the checkpoint dictionary
    checkpoint = torch.load('checkpoint.pth', weights_only=False)

    # 3. Restore states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # 4. Set to training mode
    model.eval()
    return (model,)


@app.cell
def _(evaluate_model, model, val_dataloader):
    results = evaluate_model(model=model, dataset=val_dataloader)
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(plt, results):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    event_happened =  (results["gt"] >= 0.5)
    event_predicted = (results["pred"] >= 0.5)
    cm = confusion_matrix(event_happened, event_predicted)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    return (event_predicted,)


@app.cell
def _(event_predicted, np):
    np.sum(event_predicted)# - len(event_predicted)
    return


@app.cell
def _(np, results):
    print(np.min(results["pred"]))
    print(np.max(results["pred"]))
    return


if __name__ == "__main__":
    app.run()
