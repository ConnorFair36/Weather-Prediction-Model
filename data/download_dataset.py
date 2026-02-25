import cdsapi
import xarray as xr
import numpy as np
import dask
import zarr
import gribapi

import calendar
import datetime
import yaml

# read the yaml config file dataset settings
with open('../configs/config.yaml', 'r') as file:
    config_data = yaml.safe_load(file)


config_data = config_data['data']
date_range = config_data['time']

client = cdsapi.Client()
grib_files = []
# create a new request for each year
for year in range(date_range['start']['year'], date_range['end']['year'] + 1):
    # generate the list of months for the current year
    months = []
    if year == date_range['start']['year'] and year == date_range['end']['year']:
        months = list(range(date_range['start']['month'], date_range['end']['month'] + 1))
    elif year == date_range['start']['year']:
        months = list(range(date_range['start']['month'], 13))
    elif year == date_range['end']['year']:
        months = list(range(1, date_range['end']['month'] + 1))
    else:
        months = list(range(1,13))
    # convert numbers to strings
    months = [f"{month:02d}" for month in months]

    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": config_data['variables'],
        "year": [str(year)],
        "month": months,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "grib",
        "download_format": "unarchived",
        "area": config_data['area']
    }

    
    grib_files.append(client.retrieve(dataset, request).download())
    print(f"The Dataset: {grib_files[-1]}")

# clean and save the .grib dataset as a chunked .zarr dataset
for grib_file in grib_files:
    # grab total precipitation seperatly due to the different way it handles time
    ds1 = xr.open_dataset(grib_file, engine="cfgrib", backend_kwargs={'filter_by_keys': {'shortName': 'tp'}})
    # combine time and step into a single variable
    ds_stacked = ds1.assign_coords(
        valid_time=ds1["time"] + ds1["step"]
    )
    # compress time & step into 1 dimention
    ds_stacked = ds_stacked.stack(t=("time", "step"))
    # remove all entries where all values are nan
    ds_stacked = ds_stacked.dropna(dim="t", how="all")
    # remove the old time and step dimentions and replace them with the correct time dimention
    ds_clean = (
        ds_stacked
        .set_index(t="valid_time")
        .rename({"t": "valid_time"})
        .drop_vars(["time", "step"])
        .sortby("valid_time")
        .rename({"valid_time": "time"})
    )
    # get the rest of the variables from the dataset
    ds2 = xr.open_dataset(grib_file, engine="cfgrib",
                           backend_kwargs={'filter_by_keys': {'shortName': ['10u', '10v', '2t', 'sp', 'tcc']}})
    # combine both datasets accross their shared dimentions (time, latitude, longitude)
    full_ds = xr.merge([ds_clean, ds2])
    # let dask estimate the optimal chunking strategy
    full_ds = full_ds.chunk("auto")
    # save to a .zarr file
    zarr_file_name = f"era5_conus_{date_range['start']['year']}_{date_range['start']['month']}_to_{date_range['end']['year']}_{date_range['end']['month']}_{len(config_data['variables'])}var.zarr"
    full_ds.to_zarr(
        zarr_file_name,
        mode="w",
        consolidated=True
    )