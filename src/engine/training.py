import torch
from torch import nn, utils, optim
import numpy as np
import itertools
import math


def train_model(model: nn.Module, 
                training_dataset: utils.data.Dataset,
                validation_dataset: utils.data.Dataset,
                epochs: int, 
                loss_fun: nn.Module, 
                optimizer: optim.Optimizer,
                val_freq: int = 5) -> list[list[int]]:
    """Trains to model on the given dataset and returns the training and validation loss over training."""
    training_losses = [[],[]]
    device = str(next(model.parameters()).device)
    for epoch in range(epochs):
        all_losses = [[],[]]
        # training loop
        model.train()
        for sample, truth in training_dataset:
            # get the total precipitation for the ground truth
            truth = truth[:,:,:,:,3].to(device, dtype=torch.float32)
            sample = sample.permute((0,1,4,2,3)).to(device, dtype=torch.float32)
            # adjust model weights based on loss
            pred_rain = model(sample)
            train_loss = loss_fun(pred_rain, truth)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            all_losses[0].append(train_loss.item())
            # get the validation loss for this epoch
        # validation loss
        if epoch % val_freq == 0:
            model.eval()
            for sample, truth in validation_dataset:
                # get the total precipitation for the ground truth
                truth = truth[:,:,:,:,3].to(device, dtype=torch.float32)
                sample = sample.permute((0,1,4,2,3)).to(device, dtype=torch.float32)

                with torch.no_grad():
                    pred_rain = model(sample)
                    val_loss = loss_fun(pred_rain, truth)
                    all_losses[1].append(val_loss.item())
        else:
            all_losses[1].append(np.nan)
        # aggregate the losses for the train and validation loops
        training_losses[0].append(np.mean(all_losses[0]))
        training_losses[1].append(np.mean(all_losses[1]))
    return training_losses

# functions for sucessive halving

def _get_params(parameters: dict) -> tuple[dict, dict, dict]:
    """Gets the hyperparameters for the model and optimizer"""
    model_params = dict()
    optim_params = dict()
    other = dict()
    for key, value in parameters.items():
        if key.startswith("model_"):
            model_params[key[6:]] = value
        elif key.startswith("optim_"):
            optim_params[key[6:]] = value
        else:
            other[key] = value
    return (model_params, optim_params, other)


def _train_model(model, training_dataset, budget, loss_fun, optimizer):
    device = str(next(model.parameters()).device)
    model.train()
    for epoch in range(budget):
        all_losses = [[],[]]
        # training loop
        for sample, truth in training_dataset:
            # get the total precipitation for the ground truth
            truth = truth[:,:,:,:,3].to(device, dtype=torch.float32)
            sample = sample.permute((0,1,4,2,3)).to(device, dtype=torch.float32)
            # adjust model weights based on loss
            pred_rain = model(sample)
            train_loss = loss_fun(pred_rain, truth)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()


def _val_model(model, validation_dataset, loss_fun) -> float:
    device = str(next(model.parameters()).device)
    model.eval()
    total_loss = 0
    for sample, truth in validation_dataset:
        # get the total precipitation for the ground truth
        truth = truth[:,:,:,:,3].to(device, dtype=torch.float32)
        sample = sample.permute((0,1,4,2,3)).to(device, dtype=torch.float32)
        
        with torch.no_grad():
            pred_rain = model(sample)
            total_loss += loss_fun(pred_rain, truth).item()
    return total_loss

# Source: https://arxiv.org/abs/1502.07943
def sucessive_halving(model_class: nn.Module, 
                      training_dataset: utils.data.Dataset,
                      validation_dataset: utils.data.Dataset,
                      epochs: int, 
                      hyper_parameters: dict):
    """Applies the sucessive halving algorithm to find the best hyper parameters efficiently"""
    # gets all of the possible combinations of hyper parameters
    combinations = list(itertools.product(*list(hyper_parameters.values())))
    n = len(combinations)
    if not (n > 0 and n.bit_count() == 1):
        print("the total number of combinations must be a power of 2")
        return None
    results = [[] for _ in range(n)]
    in_race = [True for _ in range(n)] # which parameter combinations are still active
    budgets =  [math.ceil(epochs/(2**(i+1))) for i in range(math.ceil(math.log2(n)), -1, -1)]
    model_checkpoints = [f"model_{idx}.pth" for idx in range(n)]
    # loop through all of the filtering rounds
    for k in range(int(math.log2(n)) + 1):
        budget = budgets[k] if k == 0 else budgets[k] - budgets[k-1]
        # loop through all of the combinations that are still being tested
        for model_idx in range(n):
            if not in_race[model_idx]:
                # sentinel value -1 assumes loss function is always non-negative
                results[model_idx].append(-1)
                continue
            # load in the model if it needs to continue training
            combo = dict(zip(hyper_parameters.keys(), combinations[model_idx]))
            model_params, optim_params, other = _get_params(combo)
            model = model_class(**model_params)
            optimizer = other["optimizer"](model.parameters(), **optim_params)
            loss = other["loss"]()
            # load in saved model and optimizer states
            if k > 0:
                checkpoint = torch.load(model_checkpoints[model_idx])
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # compile the model to boost performance
            model = torch.compile(model)
            # continue training
            _train_model(model, training_dataset, budget, loss, optimizer)
            # save model performance after training
            results[model_idx].append(_val_model(model, validation_dataset, loss))
            # update checkpoint
            checkpoint = {
                'model_state_dict': model._orig_mod.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, model_checkpoints[model_idx])
            print(f"---Finished Model {model_idx} on {k} Current loss: {results[model_idx][-1]}---")
        if k == int(math.log2(n)):
            continue
        # pick which models will not continue
        n_eliminated = sum(in_race) // 2
        current_results = [i[-1] for i in results]
        for _ in range(n_eliminated):
            worst = current_results.index(max(current_results))
            current_results[worst] = -1
            in_race[worst] = False
    
    final_results = [i[-1] for i in results]
    best_model = final_results.index(max(final_results))
    print(f"Best Model: model_{best_model}\n  Error:{final_results[best_model]}")
    




