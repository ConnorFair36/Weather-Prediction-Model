import torch
import torch.nn as nn

from .convLSTM_parts import *

class baseline_ConvLSTM(nn.Module):
    def __init__(self,
                input_dims: int, 
                hidden_dim: int | list[int],
                kernel_size: int | tuple[int, int] | list[int | tuple[int, int]],
                num_layers: int,
                num_pred_steps: int,
                batch_first: bool = True):
        super().__init__()
        self.encoder = ConvLSTM(input_dims,
                            hidden_dim,
                            kernel_size,
                            num_layers,
                            batch_first=batch_first,
                            return_all_layers=True)
        self.decoder = ConvLSTM(1,
                            hidden_dim,
                            kernel_size,
                            num_layers,
                            batch_first=batch_first,
                            return_all_layers=True)
        self.forecaster = nn.Conv2d(
            in_channels=hidden_dim * 2 * num_layers,
            out_channels=num_pred_steps,
            kernel_size=1
        )
        self.num_pred_steps = num_pred_steps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encodes the previous timesteps
        input_shapes = x.shape
        _, encoder_states =  self.encoder(x)
        # decodes the encoded values for the previous timesteps
        dummy_input = torch.zeros((input_shapes[0], self.num_pred_steps, 1, input_shapes[3], input_shapes[4]),device=x.device)
        _, decoder_states = self.decoder(dummy_input, encoder_states)
        # converts the hidden statees into predictions for the future timesteps
        decoder_states = [item for sublist in decoder_states for item in sublist]
        uncompressed_predictions = torch.cat(decoder_states, dim=1)
        output = self.forecaster(uncompressed_predictions)
        return torch.clamp(output, min=0)