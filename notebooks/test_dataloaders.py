import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os

    # Walk up one level from notebooks/ to reach the project root
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    return (os,)


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import seaborn as sns
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import imageio.v2 as imageio

    import torch
    from torch import utils, nn
    import torch.nn.functional as F

    from src.data.read_data import WeatherTrainingData, transformations
    return (
        WeatherTrainingData,
        ccrs,
        cfeature,
        imageio,
        nn,
        np,
        plt,
        torch,
        utils,
    )


@app.cell
def _(WeatherTrainingData):
    training_data = WeatherTrainingData(dir="./data/era5_conus_2025_1_to_2025_12_6var.zarr/",
                                             set_type="train",
                                             seq_length=6)
    return (training_data,)


@app.cell
def _(training_data, utils):

    training_dataloader = utils.data.DataLoader(training_data, batch_size=128, num_workers=2)

    test_data = iter(training_dataloader)
    example_data = next(test_data)
    return (example_data,)


@app.cell
def _():
    import math
    (math.ceil(((8760*0.8) // 2) / 128) *  (0+(8/60))) * 50
    return


@app.cell
def _(training_data, utils):
    import time
    time_points = [[] for _ in range(5)]
    for workers in range(1,5):
        test_training_dataloader = utils.data.DataLoader(training_data, batch_size=128, num_workers=workers)
        time_points[workers-1].append(time.monotonic())
        for data in test_training_dataloader:
            time_points[workers-1].append(time.monotonic())
    return (time_points,)


@app.cell
def _(np, time_points):
    matrix = np.array(time_points[:4])
    results = matrix[:,1:] - matrix[:,:-1]
    np.mean(results,axis=1)
    return (results,)


@app.cell
def _(np, results):
    np.sum(results,axis=1)
    return


@app.cell
def _(np, results):
    np.std(results,axis=1)
    return


@app.cell
def _(example_data):
    example_data[0][0,:,0,0,:].shape
    return


@app.cell
def _(example_data, plt):
    plt.imshow(example_data[0][0,:,1,0,:].permute((1,0)) - 273.15)
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(example_data, plt):
    plt.imshow(example_data[0][0,:,1,0,:].permute((1,0)) - 273.15)
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(ccrs, cfeature, example_data, plt):
    ax = plt.axes(projection=ccrs.Miller())
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.NaturalEarthFeature(
        category='physical',
        name='lakes',
        scale='50m',
        facecolor='none',  # No fill
        edgecolor='black'   # Boundary color
    ), linestyle=':')
    ax.add_feature(cfeature.BORDERS, linestyle='--')
    ax.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())
    plt.imshow(example_data[0][0,:,3,0,:].permute((1,0)),
              transform=ccrs.PlateCarree(),
              extent=[-125, -66.5, 24.5, 49.5],
              origin="upper")
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(ccrs, cfeature, example_data, imageio, os, plt, torch):
    # create the gif frames
    var_pos = 4
    title = "Eastward Windspeed (m/s)"
    frames_dir = './frames/'
    os.makedirs(frames_dir, exist_ok=True)

    color_min = torch.min(example_data[0][0,:,var_pos,:,:])
    color_max = torch.max(example_data[0][0,:,var_pos,:,:])


    filenames = []
    for i in range(6):
        # Create plot
        ax2 = plt.axes(projection=ccrs.Miller())
        ax2.add_feature(cfeature.LAND)
        ax2.add_feature(cfeature.OCEAN)
        ax2.add_feature(cfeature.COASTLINE)
        ax2.add_feature(cfeature.NaturalEarthFeature(
            category='physical',
            name='lakes',
            scale='50m',
            facecolor='none',  # No fill
            edgecolor='black'   # Boundary color
        ), linestyle=':')
        ax2.add_feature(cfeature.BORDERS, linestyle='--')
        ax2.set_extent([-125, -66.5, 24.5, 49.5], crs=ccrs.PlateCarree())
        plt.imshow(example_data[0][0,:,var_pos,i,:].permute((1,0)),
              transform=ccrs.PlateCarree(),
              extent=[-125, -66.5, 24.5, 49.5],
              origin="upper",
              vmin=color_min, 
              vmax=color_max,
              cmap="seismic")
        plt.colorbar(shrink=0.75)
        plt.title(title)

        # Save the figure with a unique filename
        filename = f'{frames_dir}frame_{i:02d}.png'
        filenames.append(filename)
        plt.savefig(filename)
        plt.close() # Close the figure to free memory

    images = [imageio.imread(filename) for filename in filenames]

    # Save as a GIF
    imageio.mimsave(f'animation_u_ws.gif', images, duration=400, loop=0) # duration in ms, loop=0 means infinite loop
    return (var_pos,)


@app.cell
def _(example_data, var_pos):
    example_data[0][0,:,var_pos,0,:].permute((1,0))
    return


@app.cell
def _(ccrs, example_data, imageio, nn, np, plt):
    from cartopy.feature import NaturalEarthFeature, COLORS

    full_speed = np.sqrt((example_data[0][0,:,4,:,:].permute((2,0,1)).numpy() ** 2) +
                        (example_data[0][0,:,5,:,:].permute((2,0,1)).numpy() ** 2))

    color_min2 = np.min(full_speed)
    color_max2 = np.max(full_speed)

    filenames2 = []
    for i in range(6):
        avgpool = nn.AvgPool2d(kernel_size=4, stride=4)
        u = avgpool(example_data[0][0,:,4:5,i,:].permute((1,2,0))).numpy()[0,:,:]
        v = avgpool(example_data[0][0,:,5:6,i,:].permute((1,2,0))).numpy()[0,:,:]
    
        #u = example_data[0][0,:,4,0,:].permute((1,0)).numpy()
        #v = example_data[0][0,:,5,0,:].permute((1,0)).numpy()
    
        n, m = u.shape
        lon = np.linspace(-125, -66.5, m)
        lat = np.linspace(24.5, 49.5, n)
        # Create a grid of lon/lat points
        lon, lat = np.meshgrid(lon, lat)
        # Calculate wind speed (magnitude) for optional coloring
        speed = np.sqrt(u**2 + v**2)
    
        u = u / speed
        v= v / speed
    
        fig = plt.figure(figsize=(10, 5))
        ax3 = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree()) # Use PlateCarree for simple lon/lat data
    
        # plot wind speed not downsampled
    
        wind_speed_map = plt.imshow(full_speed[:,:,i],
                  transform=ccrs.PlateCarree(),
                  extent=[-125, -66.5, 24.5, 49.5],
                  origin="upper",
                  vmin=color_min2, 
                  vmax=color_max2,
                  cmap="plasma")
    
        # Add map features
        ax3.coastlines(resolution='110m',color="darkgray")
        ax3.set_title('Wind Speed and Direction on Map')
    
        # Plot wind vectors using quiver
        # The transform=ccrs.PlateCarree() is crucial as it tells Cartopy the data's coordinate system
        # C=speed colors the arrows by wind speed magnitude
        quiver_plot = ax3.quiver(lon, lat, u, v, #speed,
                                transform=ccrs.PlateCarree(),
                                #cmap='plasma', 
                                scale=1,
                                scale_units='xy')
    
        # Add a color bar for wind speed
        fig.colorbar(wind_speed_map, ax=ax3, label='Wind Speed (m/s)')
    
        filename2 = f'wind_frame_{i:02d}.png'
        filenames2.append(filename2)
        plt.savefig(filename2)
        plt.close() # Close the figure to free memory

    images2 = [imageio.imread(filename2) for filename2 in filenames2]

    # Save as a GIF
    imageio.mimsave(f'animation_wind.gif', images2, duration=400, loop=0) # duration in ms, loop=0 means infinite loop
    return


if __name__ == "__main__":
    app.run()
