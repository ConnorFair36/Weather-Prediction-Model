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
    from src.engine.error_func import WeightedMSELoss
    import pandas as pd
    return (
        WeatherTrainingData,
        WeightedMSELoss,
        baseline_ConvLSTM,
        create_transform_function,
        evaluate_model,
        optim,
        pd,
        plt,
        sns,
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
def _(WeightedMSELoss, baseline_ConvLSTM, optim):
    model = baseline_ConvLSTM(
        input_dims=6,
        hidden_dim=20,
        kernel_size=5,
        num_layers=2,
        num_pred_steps=3
    )
    loss_fun = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    epochs = 30
    return epochs, loss_fun, model, optimizer


@app.cell
def _(
    epochs,
    loss_fun,
    model,
    optimizer,
    train_model,
    training_dataloader,
    val_dataloader,
):
    results = train_model(
        model=model,
        training_dataset=training_dataloader,
        validation_dataset=val_dataloader,
        epochs=epochs,
        loss_fun=loss_fun,
        optimizer=optimizer
    )
    return (results,)


@app.cell
def _(results):
    results
    return


@app.cell
def _(plt, results, sns):
    sns.lineplot(results[0])
    sns.lineplot(results[1])
    plt.savefig("./src/weights/trial3_trainval_curve.png")
    plt.show()
    return


@app.cell
def _(pd, results):
    df = pd.DataFrame({'train': results[0], 'validation': results[1]})
    df.to_csv("./src/weights/trial_3_trainval.csv")
    return


@app.cell
def _(evaluate_model, model, val_dataloader):
    evaluation = evaluate_model(model, val_dataloader)
    return (evaluation,)


@app.cell
def _(evaluation):
    evaluation
    return


@app.cell
def _(evaluation, plt):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    event_happened = (evaluation["gt"] >= 0.5)
    event_predicted = (evaluation["pred"] >= 0.5)
    cm = confusion_matrix(event_happened, event_predicted)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()
    return


@app.cell
def _(val_dataloader):
    val_iter = iter(val_dataloader)
    sample, truth = next(val_iter)
    return sample, truth


@app.cell
def _(sample, truth):
    print(sample.shape)
    print(truth.shape)
    return


@app.cell
def _(model, sample, torch):
    model.eval()
    with torch.no_grad():
        pred = model(sample.permute((0,1,4,2,3)).to("cpu", dtype=torch.float32))
    return (pred,)


@app.cell
def _(plt, pred, torch, truth):
    scaled_truth = torch.expm1(truth[0,:,:,:,3]) / 1_000
    scaled_pred  = torch.expm1(pred[0,:]) / 1_000

    max_rain = max(torch.max(scaled_truth), torch.max(scaled_pred))
    fig, axs = plt.subplots(2, 3,  figsize=(15, 10))

    # Access specific plots using indexing
    images = []
    rows = ["Ground\nTruth", "Prediction"]
    for i in range(3):
        im = axs[0,i].imshow(scaled_truth[i].permute((1,0)), vmin=0, vmax=max_rain)
        images.append(im)
    for i in range(3):
        axs[1,i].imshow(scaled_pred[i].permute((1,0)), vmin=0, vmax=max_rain)

    for ax, row in zip(axs[:, 0], rows):
        ax.set_ylabel(row, rotation=0, size='large', labelpad=30)

    cbar_ax = fig.add_axes([0.93, 0.345, 0.015, 0.3]) 
    cbar = fig.colorbar(images[2], cax=cbar_ax, orientation='vertical', fraction=0.02, pad=0.04)
    cbar_ax.set_title("Rain (m)", pad=12)
    plt.subplots_adjust(wspace=0.2, hspace=-0.7)
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16, y=0.73)
    plt.savefig("./src/weights/trial_3_sample.png")
    plt.show()
    return


@app.cell
def _(plt, pred):
    plt.imshow(pred[0,0].permute((1,0)))
    return


@app.cell
def _(plt, truth):
    plt.imshow(truth[0,0,:,:,3].permute((1,0)))
    return


@app.cell
def _(model, optimizer, torch):
    checkpoint = {
        'epoch': 30,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, './src/weights/BaselineLSTM_Trial_3.pth')
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
