import xarray as xr
import numpy as np
import dask
import zarr
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import json

from sklearn.model_selection import train_test_split
from scipy.stats import boxcox

PREPROCESSING_VARS = "./src/data/preprocess_vars.json"

class WeatherTrainingData(Dataset):
    def __init__(self, dir: str, set_type: str, seq_length=6, gt_len=3, transform=None, splits=(0.8, 0.1, 0.1), seed=1339, n_months=12):
        """Takes the path to the zarr dataset, sequence length and if you want the test, train or validation set as input."""
        # yell at the user for being wrong
        if set_type not in ["train", "test", "validation"]:
            raise ValueError("set_type must be one of the following: [train, test, validation]")
        if gt_len > seq_length or gt_len <= 0:
            raise ValueError("ground truth length must be shorter than the sequence length and greater than 0")
        # save the seed used for the dataloader to help with mapping values for preprocessing to the used data split
        self.seed = seed
        # number of values that the model tries to predict
        self.gt_len = gt_len
        self.weather_dataset = xr.open_dataset(dir, engine="zarr")
        # sequence length should be divisible by the total length
        self.seq_length = seq_length
        self.transform = transform
        # create the test, train and validation splits evenly
        length = self.weather_dataset.sizes["time"] // self.seq_length
        all_indexes = list(range(length))
        # aproximates the months to ensure even distribution of samples
        month_aprx = [i % length//n_months for i in range(length)]

        # split between training and test/validation
        train_idx, other_idx, _, month_aprx = train_test_split(all_indexes, month_aprx, train_size=splits[0], stratify=month_aprx, random_state=seed)
        # split between test and validation
        test_idx, validation_idx = train_test_split(other_idx, train_size=splits[1]/(splits[1] + splits[2]), stratify=month_aprx, random_state=seed)

        if set_type == "test":
            self.indexes = test_idx
        elif set_type == "validation":
            self.indexes = validation_idx
        else:
            self.indexes = train_idx
        

    def __len__(self):
        """Returns the length of the dataset - the sequence length, which is the number of groups of batches stored."""
        return len(self.indexes)

    def __getitem__(self, idx):
        """Gets the data stored for a particular hour with the dimentions (latitude, longitude, time step, weather variables)."""
        # translate the inputed index to the index for the dataset
        idx = self.indexes[idx]
        # gets the subset of the dataset we are looking for

        sample_dims = ['latitude', 'longitude', 'time']
        # convert the substep from dataset to -> stacked array -> numpy -> torch tensor
        stacked_array = self.weather_dataset.isel(time=range(idx * self.seq_length, (idx+1) * self.seq_length)).to_stacked_array("valid_time", sample_dims=sample_dims)
        numpy_array = stacked_array.to_numpy()
        # get the min and max for normalization here because torch min and max returns nan
        sample = torch.from_numpy(numpy_array)
        # the sample needs to be reshaped into:
        #   (timestep, longitude, latitude, "image")
        sample = sample.permute(0,2,1,3).contiguous()
        # do a transformation of the final tensor if applicable
        if self.transform:
            sample = self.transform(sample)
        # move the last timestep into a seperate tensor and keep only the precipitation
        truth = sample[-self.gt_len:,:,:,:]
        sample = sample[:-self.gt_len,:,:,:]
        return sample, truth
    
    def get_all_indexes(self):
        """Returns all of the indexes being used for whatever dataset type was chosen"""
        all_indexes = []
        for idx in self.indexes:
            all_indexes.append(list(range(idx * self.seq_length, (idx+1) * self.seq_length)))
        return [item for sublist in all_indexes for item in sublist]


def _get_training_data(dir: str, seed: int=1339) -> xr.Dataset:
    """Gets the training data as an xarray dataset."""
    temp_dataset = WeatherTrainingData(dir, "train", seed=seed)
    training_indexes = temp_dataset.get_all_indexes()
    original_dataset = temp_dataset.weather_dataset
    return original_dataset.isel(time=training_indexes)


def generate_transforms(dir: str, lda: list[str], mean: list[str], std: list[str], min: list[str], max: list[str], seed: int=1339):
    """Generates values needed for preprocessing based on the training set, on the specified variables such as the
     mean or standard deviation and saves them to a json file."""
    np.random.seed(seed)
    # get the training set as an xarray dataset
    training_dataset = _get_training_data(dir, seed=seed)
    # get all of the existing data from the json file
    try:
        with open(PREPROCESSING_VARS, 'r+') as file:
            # Load the file content into a Python list/dict
            file_data = json.load(file)
    except FileNotFoundError:
        # If the file doesn't exist, create it
        with open(PREPROCESSING_VARS, 'w') as file:
            json.dump(dict(), file)
            file_data = dict()
    # create variables for each entry on this seed if it doesn't exist yet
    if not file_data.get(str(seed), False):
        file_data[str(seed)] = {
            "lambda": dict(),
            "mean": dict(),
            "std": dict(),
            "min": dict(),
            "max": dict(),
        }
    # estimate the lambda value due to the large size of the dataset
    for var in lda:
        if file_data[str(seed)]["lambda"].get(var, None) == None:
            sample = np.random.choice(training_dataset[var].values.flatten(), size=100_000, replace=False,)
            _, file_data[str(seed)]["lambda"][var] = boxcox(sample)
    
    for var in mean:
        if file_data[str(seed)]["mean"].get(var, None) == None:
            file_data[str(seed)]["mean"][var] = float(training_dataset[var].mean())
    
    for var in std:
        if file_data[str(seed)]["std"].get(var, None) == None:
            file_data[str(seed)]["std"][var] = float(training_dataset[var].std())
    
    for var in min:
        if file_data[str(seed)]["min"].get(var, None) == None:
            file_data[str(seed)]["min"][var] = float(training_dataset[var].min())

    for var in max:
        if file_data[str(seed)]["max"].get(var, None) == None:
            file_data[str(seed)]["max"][var] = float(training_dataset[var].max())
    
    # save the variables to the json file
    with open(PREPROCESSING_VARS, 'w') as file:
            json.dump(file_data, file)


def create_transform_function(dir: str, transforms: dict, seed: int=1339):
    """Creates a function that takes a tensor as input and applies the transformations to the variables stored in it."""
    # generate the transformations
    gen_values = {
        "lambda": [],
        "mean": [],
        "std": [],
        "min": [],
        "max": []
    }
    for name in ["sp", "t2m", "tcc", "tp", "u10", "v10"]:
        for t in transforms.get(name, []):
            if t == "z-scale":
                gen_values["mean"].append(name)
                gen_values["std"].append(name)
            elif t == "-1to1":
                gen_values["min"].append(name)
                gen_values["max"].append(name)
            elif t == "boxcox":
                gen_values["lambda"].append(name)
            elif t == "0to1":
                gen_values["min"].append(name)
                gen_values["max"].append(name)

    generate_transforms(dir, lda=gen_values["lambda"], mean=gen_values["mean"], std=gen_values["std"], min=gen_values["min"], max=gen_values["max"])
    # get values from the json file 
    with open(PREPROCESSING_VARS, 'r+') as file:
        # Load the file content into a Python list/dict
        file_data = json.load(file)
    transformation_values = file_data[str(seed)]
    def transformation(data: torch.tensor) -> torch.tensor:
        # data variables are in the order: sp, t2m, tcc, tp, u10, v10
        for idx, name in enumerate(["sp", "t2m", "tcc", "tp", "u10", "v10"]):
            for t in transforms.get(name,[]):
                if t == "m-to-mm":
                    data[:,:,:,idx] *= 1_000
                elif t == "logp1":
                    data[:,:,:,idx] = data[:,:,:,idx].log1p()
                elif t == "z-scale":
                    data[:,:,:,idx] = (data[:,:,:,idx] - transformation_values["mean"][name]) / transformation_values["std"][name]
                elif t == "-1to1":
                    data[:,:,:,idx] = data[:,:,:,idx] / max(abs(transformation_values["min"][name]),transformation_values["max"][name])
                elif t == "0to1":
                    data[:,:,:,idx] = (data[:,:,:,idx] - transformation_values["min"][name]) / (transformation_values["max"][name] - transformation_values["min"][name])
                elif t == "boxcox":
                    l = transformation_values["lambda"][name]
                    if l != 0:
                        data[:,:,:,idx] = ((data[:,:,:,idx] ** l) - 1) / l
                    else:
                        data[:,:,:,idx] = data[:,:,:,idx].log()
        return data
    return transformation
