import numpy as np
import torch
from torch import nn

from sklearn.metrics import root_mean_squared_error, confusion_matrix

def correlation(ground_truth: np.array, predictions: np.array) -> float:
    # correlation comes from the metric defined on page 7 of: https://arxiv.org/pdf/1506.04214
    numerator = np.sum(ground_truth * predictions)
    denominator = np.sum(np.square(ground_truth)) * np.sum(np.square(predictions)) + 1e-9
    return numerator / denominator

def critical_success_index(cm: np.array) -> float:
    hits = cm[1, 1]
    misses = cm[1, 0]
    false_alarms = cm[0, 1]
    return hits / (hits + misses + false_alarms)

def false_alarm_ratio(cm: np.array) -> float:
    hits = cm[1, 1]
    false_alarms = cm[0, 1]
    return false_alarms / (hits + false_alarms)

def probability_of_detection(cm: np.array) -> float:
    hits = cm[1, 1]
    misses = cm[1, 0]
    return hits / (hits + misses)

def evaluate_model(model: nn.Module, dataset: torch.utils.data.dataloader.DataLoader) -> dict:
    """Returns a dictionary containing all of the values used for evaluating model performance based on the given dataset."""
    # get the device that the model is currently on
    device = model.parameters().__next__().device
    # loop through the evaluation dataset and save predictions
    predictions = []
    ground_truth = []
    for sample, truth in dataset:
        # get the total precipitation for the ground truth
        ground_truth.append(truth[:,:,:,:,3].detach().cpu().numpy())
        sample = sample.permute((0,1,4,2,3)).to(device, dtype=torch.float32)
        # predict rain for current batch
        with torch.no_grad():
            pred_rain = model(sample)
        # save prediction
        predictions.append(pred_rain.detach().cpu().numpy())
    # combine all predictions and ground truths into 1 numpy array
    ground_truth = np.concat(ground_truth, axis=0)
    predictions = np.concat(predictions, axis=0)
    # all of the rain has been scaled log-scaled, so this must be removed before evaluation
    ground_truth = np.expm1(ground_truth).flatten()
    predictions = np.expm1(predictions).flatten()

    # regression metrics
    rmse = root_mean_squared_error(ground_truth, predictions)
    corr = correlation(ground_truth, predictions)
    # meterological metrics
    #   the event being measured is if rainfall >= 0.5 mm is detected
    event_happened = (ground_truth >= 0.5)
    event_predicted = (predictions >= 0.5)
    cm = confusion_matrix(event_happened, event_predicted)
    csi = critical_success_index(cm)
    far = false_alarm_ratio(cm)
    pod = probability_of_detection(cm)
    return {
        "rmse": rmse,
        "corr": corr,
        "csi": csi,
        "far": far,
        "pod": pod,
        "gt": ground_truth,
        "pred": predictions
    }
