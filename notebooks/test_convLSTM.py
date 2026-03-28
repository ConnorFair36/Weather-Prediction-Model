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
    from torch import nn, utils
    from src.models.convLSTM_parts import ConvLSTM
    from src.engine.training import train_model

    import matplotlib.pyplot as plt
    import seaborn as sns

    from src.data.read_data import WeatherTrainingData, create_transform_function
    return (
        ConvLSTM,
        WeatherTrainingData,
        create_transform_function,
        nn,
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
def _(ConvLSTM, nn, torch):
    class baseline_ConvLSTM(nn.Module):
        def __init__(self,
                    input_dims: int, 
                    hidden_dim: int | list[int],
                    kernel_size: int | tuple[int, int] | list[int | tuple[int, int]],
                    num_layers: int,
                    num_pred_steps: int,
                    batch_first: bool = True):
            super().__init__()
            self.encoder = ConvLSTM(input_dims,
                                hidden_dim,
                                kernel_size,
                                num_layers,
                                batch_first=batch_first,
                                return_all_layers=True)
            self.decoder = ConvLSTM(1,
                                hidden_dim,
                                kernel_size,
                                num_layers,
                                batch_first=batch_first,
                                return_all_layers=True)
            self.forecaster = nn.Conv2d(
                in_channels=hidden_dim * 2 * num_layers,
                out_channels=num_pred_steps,
                kernel_size=1
            )
            self.num_pred_steps = num_pred_steps
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # encodes the previous timesteps
            input_shapes = x.shape
            _, encoder_states =  self.encoder(x)
            # decodes the encoded values for the previous timesteps
            dummy_input = torch.zeros((input_shapes[0], self.num_pred_steps, 1, input_shapes[3], input_shapes[4]),device=x.device)
            _, decoder_states = self.decoder(dummy_input, encoder_states)
            # converts the hidden statees into predictions for the future timesteps
            decoder_states = [item for sublist in decoder_states for item in sublist]
            uncompressed_predictions = torch.cat(decoder_states, dim=1)
            output = self.forecaster(uncompressed_predictions)
            return output
    return (baseline_ConvLSTM,)


@app.cell
def _(baseline_ConvLSTM):
    model = baseline_ConvLSTM(6, 12, 3, 2, 3)
    return (model,)


@app.cell
def _(training_dataloader):
    temp = iter(training_dataloader)
    sample_test, truth_test = next(temp)
    return (sample_test,)


@app.cell
def _(sample_test):
    sample_test.shape
    return


@app.cell
def _(model, sample_test, torch):
    model.eval()
    with torch.no_grad():
        output = model(sample_test.permute((0,1,4,2,3)))
    return (output,)


@app.cell
def _(output):
    output.shape
    return


@app.cell
def _(model, train_model, training_dataloader, val_dataloader):
    results = train_model(
        model=model,
        training_dataset=training_dataloader,
        validation_dataset=val_dataloader,
        device="mps",
        lr=1e-4,
    
    )
    return


if __name__ == "__main__":
    app.run()
