import torch
from torch import nn, utils, optim
import numpy as np

def train_model(model: nn.Module, 
                training_dataset: utils.data.Dataset,
                validation_dataset: utils.data.Dataset,
                device: str,
                lr: float, 
                epochs: int, 
                loss_fun: nn.Module, 
                optimizer: optim.Optimizer) -> list[list[int]]:
    """Trains to model on the given dataset and returns the training and validation loss over training."""
    training_losses = [[],[]]
    for epoch in epochs:
        all_losses = [[],[]]
        # training loop
        model.train()
        for sample, truth in training_dataset:
            # get the total precipitation for the ground truth
            truth = truth[:,:,:,:,3:4].to(device, dtype=torch.float32)
            sample = sample.to(device, dtype=torch.float32)
            # adjust model weights based on loss
            pred_rain = model(sample)
            train_loss = loss_fun(pred_rain, truth)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            all_losses[0].append(train_loss.item())
            # get the validation loss for this epoch
        # validation loss
        model.eval()
        for sample, truth in validation_dataset:
            # get the total precipitation for the ground truth
            truth = truth[:,:,:,:,3:4].to(device, dtype=torch.float32)
            sample = sample.to(device, dtype=torch.float32)
            
            with torch.no_grad():
                pred_rain = model(sample)
                val_loss = loss_fun(pred_rain, truth)
                all_losses[1].append(val_loss.item())
        # aggregate the losses for the train and validation loops
        training_losses[0].append(np.mean(all_losses[0]))
        training_losses[1].append(np.mean(all_losses[1]))
    return training_losses
