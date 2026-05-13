import os
import argparse

from src.config import config

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data.read_data import WeatherTrainingData, create_transform_function
from src.models.convGRU import baseline_ConvGRU
from src.models.convLSTM import baseline_ConvLSTM
from src.engine.training import train_model_with_scaling, train_model
from src.engine.eval_model import evaluate_model, evaluate_model_by_t
from src.engine.error_func import WeightedMSELoss, SpatialLoss, LocalLoss


def get_file_arg():
    # 1. Initialize the parser
    parser = argparse.ArgumentParser(description="The main script for running the precipitation prediction model.")
    # 2. Add arguments
    parser.add_argument("cfg_file", help="The path to the config.yaml file.")
    # 3. Parse arguments
    args = parser.parse_args()
    # 4. Use the values
    return str(args.cfg_file)


def get_transforms():
    prep = config.DATASET.PREPROCESSING
    return {
        "sp":  list(prep.SP),
        "t2m": list(prep.T2M),
        "tcc": list(prep.TCC),
        "tp":  list(prep.TP),
        "u10": list(prep.U10),
        "v10": list(prep.V10),
    }


def get_dataloaders():
    var_transforms = get_transforms()
    transform = create_transform_function(config.DATASET.LOCATION, transforms=var_transforms)
    train_data = WeatherTrainingData(dir=config.DATASET.LOCATION, set_type="train",      transform=transform)
    val_data   = WeatherTrainingData(dir=config.DATASET.LOCATION, set_type="validation", transform=transform)
    test_data  = WeatherTrainingData(dir=config.DATASET.LOCATION, set_type="test",       transform=transform)
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_data,  batch_size=128, shuffle=True)
    return train_loader, val_loader, test_loader


def get_model():
    if config.MODEL.ARCHITECTURE == "ConvGRU":
        return baseline_ConvGRU(
            input_dims=config.MODEL.INPUT_DIMS,
            hidden_dim=config.MODEL.HIDDEN_DIM,
            kernel_size=config.MODEL.KERNEL_SIZE,
            num_layers=config.MODEL.NUM_LAYERS,
            num_pred_steps=config.MODEL.NUM_PRED_STEPS,
        )
    elif config.MODEL.ARCHITECTURE == "ConvLSTM":
        return baseline_ConvLSTM(
            input_dims=config.MODEL.INPUT_DIMS,
            hidden_dim=config.MODEL.HIDDEN_DIM,
            kernel_size=config.MODEL.KERNEL_SIZE,
            num_layers=config.MODEL.NUM_LAYERS,
            num_pred_steps=config.MODEL.NUM_PRED_STEPS,
        )
    else:
        print("A valid model name must be provided")
        return None


def get_optim(model):
    name_to_optim = {
        "SGD": optim.SGD,
        "SGDmomentum": optim.SGD,
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
    }
    optimizer_cls = name_to_optim[config.TRAINING.OPTIMIZER]
    if config.TRAINING.OPTIMIZER == "SGDmomentum":
        return optimizer_cls(model.parameters(), lr=config.TRAINING.LEARNING_RATE, momentum=0.9)
    return optimizer_cls(model.parameters(), lr=config.TRAINING.LEARNING_RATE)


def get_loss_fn():
    name_to_loss = {
        "WeightedMSELoss": WeightedMSELoss,
        "SpatialLoss": SpatialLoss,
        "LocalLoss": LocalLoss,
    }
    return name_to_loss[config.TRAINING.LOSS]()


def create_training_folder():
    folder_name = f"src/weights/{config.TRAINING.TEST_NAME}"
    counter = 1
    temp_name = folder_name

    while os.path.exists(temp_name):
        temp_name = f"{folder_name}_{counter}"
        counter += 1

    os.makedirs(temp_name)
    return temp_name


def training_mode():
    train_loader, val_loader, _ = get_dataloaders()

    model = get_model()
    model = torch.compile(model)
    loss_fn = get_loss_fn()
    optimizer = get_optim(model)
    folder_name = create_training_folder()

    if config.TRAINING.LOSS in {"LocalLoss"}:
        results = train_model_with_scaling(
            model=model,
            training_dataset=train_loader,
            validation_dataset=val_loader,
            epochs=config.TRAINING.EPOCHS,
            loss_fun=loss_fn,
            optimizer=optimizer,
            val_freq=1,
        )
    elif config.TRAINING.LOSS in {"WeightedMSELoss", "SpatialLoss"}:
        results = train_model(
            model=model,
            training_dataset=train_loader,
            validation_dataset=val_loader,
            epochs=config.TRAINING.EPOCHS,
            loss_fun=loss_fn,
            optimizer=optimizer,
            val_freq=1,
        )
    else:
        print("A valid loss function must be entered")
        return None
    # save model checkpoint
    checkpoint = {
        "epoch": config.TRAINING.EPOCHS,
        "model_state_dict": model._orig_mod.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{folder_name}/{config.TRAINING.TEST_NAME}.pth")

    # save loss curves
    df = pd.DataFrame({"train": results[0], "validation": results[1]})
    df.to_csv(f"{folder_name}/{config.TRAINING.TEST_NAME}.csv", index=False)

    sns.lineplot(data=df, dashes=False)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.tight_layout()
    plt.savefig(f"{folder_name}/{config.TRAINING.TEST_NAME}_loss_curve.png")


def inference_mode():
    assert config.INFERENCE.DATASET in ("validation", "test"), \
        f"INFERENCE.DATASET must be 'validation' or 'test', got '{config.INFERENCE.DATASET}'"

    _, val_loader, test_loader = get_dataloaders()
    loader = val_loader if config.INFERENCE.DATASET == "validation" else test_loader

    model = get_model()
    state = torch.load(config.INFERENCE.MODEL_WEIGHTS)
    model.load_state_dict(state["model_state_dict"])

    evaluation = evaluate_model(model, loader)

    print("Overall Metrics:")
    for key, value in evaluation.items():
        print(f"  {key}: {value:.4f}")

    evaluation_by_t = evaluate_model_by_t(model, loader)
    print("Metrics by Timestep:")
    for key, values in evaluation_by_t.items():
        print(f"  {key}:")
        for value in values:
            print(f"    {value:.4f}")


if __name__ == "__main__":
    # read in the configuration file from the cli
    cfg_file = get_file_arg()
    config.merge_from_file(cfg_file)
    config.freeze()
    # run or train the model using the set configurations
    if config.MODE == "training":
        training_mode()
    elif config.MODE == "inference":
        inference_mode()
