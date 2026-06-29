# Weather-Prediction-Model

## Introduction

Over the past 50 years, extreme weather events have been estimated to cause 2 million deaths and cause around 4 trillion USD in property damage. Being able to predict when and where these events will occur is a significant challenge, especially in developing countries that don’t have warning systems in place. 90% of the deaths and property damage in the metrics above were in developing countries. Since the 1950’s, most weather predictions have been made using expensive numerical models on supercomputers using large amounts of data. This abundance of collected data is also great for training powerful machine learning models and a more accessible alternative to the numerical models.

In this project, I took inspiration from [this paper](https://arxiv.org/abs/1506.04214) and applied their ConvLSTM to data collected from the [era5 hourly dataset](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview). I also experimented with combining the GRU and ConvLSTM architectures to improve performance and new loss functions to improve accurucy. 

[The article going over all of the experiments I ran for this project](https://connorfair36.github.io/mach-learn/weather-pred/)

## Usage

### 1. Clone the repository

```bash
git clone https://github.com/ConnorFair36/Tomato-Leaf-Disease-Prediction
cd Tomato-Leaf-Disease-Prediction
```

### 2. Set up your Python environment

This example uses [uv](https://github.com/astral-sh/uv), but any Python environment manager will work.

```bash
uv init
uv add -r requirements.txt
```

### 3. Download the dataset

The dataset is downloaded from the Climate Store API and is automaticly stored as a .grib file before being converted into a .zarr file for easier access. This requires the `eccodes` package to be installed. You can find more system-specific information [here](https://github.com/ecmwf/eccodes)

```
cd data
uv run download_dataset.py
cd ..
```

### Training

To train a model, go to the config.yaml file and set `MODE: "training"`. This will create a new folder in `src/weights` that will store the model weights, optimizer state, training and validation curve plot and csv. To run you training parameters run:

```
uv run main.py configs/config.yaml
```

### Inference

To test a model, go to the config.yaml file and set `MODE: "inference"`. This will print out the overall metrics and the metrics for each timestep. To run:

```
uv run main.py configs/config.yaml
```

## Resources

[Dataset](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview)

[LSTM Tutorial](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

[A case study of spatiotemporal forecasting techniques for weather forecasting](https://arxiv.org/abs/2209.14782v2)

[Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting](https://arxiv.org/abs/1506.04214)
