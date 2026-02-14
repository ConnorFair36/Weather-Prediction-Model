import xarray as xr
import numpy as np
import dask
import zarr
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

class WeatherTrainingData(Dataset):
    def __init__(self, dir: str, set_type: str, seq_length=6, transform=None, splits=(0.8, 0.1, 0.1), seed=42, n_months=12):
        """Takes the path to the zarr dataset, sequence length and if you want the test, train or validation set as input."""
        # yell at the user for being wrong
        if set_type not in ["train", "test", "validation"]:
            raise ValueError("set_type must be one of the following: [train, test, validation]")
        self.weather_dataset = xr.open_dataset(dir, engine="zarr")
        # sequence length should be divisible by the total length
        self.seq_length = seq_length
        self.transform = transform
        # create the test, train and validation splits evenly
        length = self.weather_dataset.sizes["valid_time"] // self.seq_length
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

        sample_dims = ['latitude', 'longitude', 'valid_time']
        # convert the substep from dataset to -> stacked array -> numpy -> torch tensor
        stacked_array = self.weather_dataset.isel(valid_time=range(idx * self.seq_length, (idx+1) * self.seq_length)).to_stacked_array("time", sample_dims=sample_dims)
        numpy_array = stacked_array.to_numpy()
        # get the min and max for normalization here because torch min and max returns nan
        sample = torch.from_numpy(numpy_array)
        # the sample needs to be reshaped into:
        #   (timestep, longitude, latitude, "image")
        sample = sample.permute(2,3,0,1).contiguous()
        # do a transformation of the final tensor if applicable
        if self.transform:
            sample = self.transform(sample)
        # move the last timestep into a seperate tensor and keep only the precipitation
        truth = sample[-1,:,:,:]
        sample = sample[:-1,:,:,:]
        return sample, truth

def transformations(data: torch.tensor, log_scale=True, to_C=True, to_polar=True):
    """The set of transformations for rescaling the data."""
    # log scale precipitation in mm
    if log_scale:
        data[:,1,:,:] = torch.log1p_(data[:,1,:,:] * 1000)
    # move temp from K to C  
    if to_C:
        data[:,0,:,:] = data[:,0,:,:] - 273.15
    # translate cartesian wind direction to polar coordinates
    #  normalizing the x-y vectors directly would lose information on direction
    if to_polar:
        # magnitude = sqrt(x^2 + y^2)
        magnitude = torch.sqrt(data[:,2,:,:]**2 + data[:,3,:,:]**2)
        # angle = arctan(x, y)
        torch.atan2(data[:,2,:,:], data[:,3,:,:], out=data[:,3,:,:])
        data[:,2,:,:] = magnitude
    return data