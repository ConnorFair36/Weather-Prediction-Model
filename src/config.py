from yacs.config import CfgNode as CN

_C = CN()

# what mode to run the system in (training, inference)
_C.MODE = "training"

# settings for the dataset
_C.DATASET = CN()

# path to the zarr dataset
_C.DATASET.LOCATION = "./era5_conus_2025_1_to_2025_12_6var.zarr"
# preprocessing pipeline steps applied to each variable
_C.DATASET.PREPROCESSING = CN()
_C.DATASET.PREPROCESSING.SP = ["0to1"]
_C.DATASET.PREPROCESSING.T2M = ["z-scale"]
_C.DATASET.PREPROCESSING.TCC = []
_C.DATASET.PREPROCESSING.TP = ["m-to-mm", "logp1"]
_C.DATASET.PREPROCESSING.U10 = ["-1to1"]
_C.DATASET.PREPROCESSING.V10 = ["-1to1"]

# settings for the model architecture
_C.MODEL = CN()

# the model architecture to use
_C.MODEL.ARCHITECTURE = "ConvGRU"
# number of input variables
_C.MODEL.INPUT_DIMS = 6
# number of hidden channels in each ConvGRU layer
_C.MODEL.HIDDEN_DIM = 20
# convolutional kernel size
_C.MODEL.KERNEL_SIZE = 5
# number of stacked ConvGRU layers
_C.MODEL.NUM_LAYERS = 2
# number of future timesteps to predict
_C.MODEL.NUM_PRED_STEPS = 3

# settings for model training
_C.TRAINING = CN()

# number of epochs to train for
_C.TRAINING.EPOCHS = 30
# the optimizer to use: [Adam, AdamW, SGDmomentum]
_C.TRAINING.OPTIMIZER = "AdamW"
# the learning rate for the optimizer
_C.TRAINING.LEARNING_RATE = 1e-4
# the loss function to use
_C.TRAINING.LOSS = "LocalLoss"
# name used for saving weights and training logs
_C.TRAINING.TEST_NAME = "test_run"

# settings for running inference
_C.INFERENCE = CN()

# path to the saved model weights to load
_C.INFERENCE.MODEL_WEIGHTS = ""
# which dataset split to run inference on (validation, test)
_C.INFERENCE.DATASET = "test"

config = _C
