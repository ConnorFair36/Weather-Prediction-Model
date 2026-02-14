import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from read_data import WeatherTrainingData, transformations

    import torch
    from torch import nn, optim
    return WeatherTrainingData, mo, nn, np, optim, plt, torch, transformations


@app.cell
def _(WeatherTrainingData, torch, transformations):
    # import the data from the dataset
    # training
    train_dataset = WeatherTrainingData(dir="./training/data/era5_conus_downsampled.zarr",set_type="train",seq_length=2,transform=transformations)
    loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128,shuffle=True)
    # validation
    validation_dataset = WeatherTrainingData(dir="./training/data/era5_conus_downsampled.zarr",set_type="validation",seq_length=2,transform=transformations)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=128,shuffle=True)
    return loader, validation_loader


@app.cell
def _(np, torch):
    # set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # pick the best device to use for the best performance
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available(): # For Apple Silicon Macs
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    device
    return (device,)


@app.cell
def _(loader):
    dataiter = iter(loader)
    images_test, truth_test = next(dataiter)
    return images_test, truth_test


@app.cell
def _(images_test, truth_test):
    print(images_test.shape)
    print(truth_test[:,1:2,:,:].shape)
    return


@app.cell
def _(nn, torch):
    class AR_CNN(nn.Module):
        def __init__(self, filters1=16, filters2=64):
            super(AR_CNN, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(4, filters1, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(filters1, filters2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(filters2, 1, kernel_size=3, padding=1)
            )

        def forward(self, x: torch.tensor):
            if x.ndim == 5:
                x = x.squeeze(dim=1)
            return self.conv_layers(x)
    return (AR_CNN,)


@app.cell
def _(AR_CNN, device, loader, nn, np, optim, plt, torch, validation_loader):
    model = AR_CNN()
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    epochs = 50
    losses = []
    val_losses = []

    model = model.to(device, dtype=torch.float32)

    for epoch in range(epochs):
        # Training phase
        epoch_train_losses = []
        model.train()  # Set model to training mode
    
        for images, truth in loader:
            images = images.to(device, dtype=torch.float32)
            truth = truth[:,1:2,:,:].to(device, dtype=torch.float32)
        
            images[images.isnan()] = -99.0
            truth_mask = ~truth.isnan()
        
            pred_rain = model(images)
            train_loss = loss_function(pred_rain[truth_mask], truth[truth_mask])
        
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
            epoch_train_losses.append(train_loss.item())
    
        # Validation phase
        model.eval()  # Set model to evaluation mode
        epoch_val_losses = []  # MOVED OUTSIDE THE LOOP
    
        for images, truth in validation_loader:
            images = images.to(device, dtype=torch.float32)
            truth = truth[:,1:2,:,:].to(device, dtype=torch.float32)
        
            images[images.isnan()] = -99.0
            truth_mask = ~truth.isnan()
        
            with torch.no_grad():
                pred_rain = model(images)
                val_loss = loss_function(pred_rain[truth_mask], truth[truth_mask])
                epoch_val_losses.append(val_loss.item())
    
        # Calculate epoch averages
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
    
        losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
    
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    plt.style.use('ggplot')
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return losses, model, val_losses


@app.cell
def _(model, torch):
    # save the model weights
    torch.save(model.state_dict(), "./training/models/AR_CNN_16_64_n1.pt")
    return


@app.cell
def _(losses, plt, val_losses):
    plt.style.use('ggplot')
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label='Traning Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("./training/models/AR_CNN_16_64_n1_trainval_loss.png")
    plt.show()
    return


@app.cell
def _(device, model, np, plt, torch, validation_loader):
    # Get a batch from validation loader
    dataiter2 = iter(validation_loader)
    images_test2, truth_test2 = next(dataiter2)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot input precipitation (channel 1 is precipitation)
    im0 = axes[0].imshow(images_test2[0, 0, 1, :, :], cmap='viridis')
    axes[0].grid(False)
    axes[0].set_title('Input Precipitation')
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    # Move to device and prepare data
    images_test2 = images_test2.to(device, dtype=torch.float32)
    truth_test2 = truth_test2[:,1:2,:,:].to(device, dtype=torch.float32)
    images_test2[images_test2.isnan()] = -99.0
    truth_mask2 = truth_test2.isnan()

    # Get model prediction
    model.eval()
    with torch.no_grad():
        rain_pred = model(images_test2)
        rain_pred[truth_mask2] = np.nan



    # Plot ground truth
    im1 = axes[1].imshow(truth_test2[0, 0, :, :].cpu(), cmap='viridis')
    axes[1].grid(False)
    axes[1].set_title('Ground Truth')
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    # Plot prediction
    im2 = axes[2].imshow(rain_pred[0, 0, :, :].cpu(), cmap='viridis')
    axes[2].set_title('Model Prediction')
    axes[2].grid(False)
    fig.colorbar(im2, ax=axes[2], shrink=0.8)

    plt.tight_layout()
    plt.savefig('1st_model_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    return images_test2, rain_pred, truth_test2


@app.cell
def _(plt, rain_pred, truth_test2):
    plt.imshow(truth_test2[0, 0, :, :].cpu() - rain_pred[0, 0, :, :].cpu(), cmap='inferno')
    plt.colorbar(shrink=0.8)
    plt.grid(False)
    plt.show()
    return


@app.cell
def _(images_test2, plt, rain_pred):
    plt.imshow(images_test2[0, 0, 1, :, :].cpu() - rain_pred[0, 0, :, :].cpu(), cmap='inferno')
    plt.colorbar(shrink=0.8)
    plt.grid(False)
    plt.show()
    return


@app.cell
def _(plt, truth_test):
    # create mask
    mask = ~truth_test.isnan()
    plt.imshow(mask[15,1,:,:])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Model Results
    Model 1:
    - Epochs: 50
    - Train Loss: 0.077370
    - Val Loss: 0.074775
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    pair-samples:
    - 64-batches:  1.3006 sec.  71 sec. total
    - 128-batches: 2.5575 sec.  71 sec. total
    - 256-batches: 5.2380 sec.  73 sec. total
    - 512-batches: 10.3914 sec. 72 sec. total
    """)
    return


@app.cell
def _():
    a = [0.1,0.2]
    print(f"{a[-1]:.6f}")
    return


if __name__ == "__main__":
    app.run()
