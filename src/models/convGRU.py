import torch
import torch.nn as nn

from typing import cast

class DepthWise_Conv2d(nn.Module):
    """A simple depth-wise convolution."""
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple[int, int],
                 padding: tuple[int, int],
                 bias: bool = True):
        super().__init__()

        self.convolutions = nn.Sequential(
            # the spatial convolution
            nn.Conv2d(in_channels,
                       in_channels, 
                       kernel_size=kernel_size, 
                       padding=padding, 
                       groups=in_channels, 
                       bias=bias),
            # depth-wise merge
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.tensor):
        return self.convolutions(x)


class ConvGRUCell(nn.Module):
    """A single ConvGRU cell module."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: tuple[int, int],
        bias: bool = True,
    ) -> None:
        """Initializes a ConvGRUCell.

        Args:
            input_dim: Number of channels of input tensor.
            hidden_dim: Number of channels of hidden state.
            kernel_size: Size of the convolutional kernel.
            bias: Whether or not to add the bias.
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.update_reset_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=2 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )
        self.update_conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

    def forward(
        self, input_tensor: torch.Tensor, cur_state: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the ConvGRUCell.

        Args:
            input_tensor: Tensor of shape (b, c, h, w).
            cur_state: The current hidden state.

        Returns:
            The next hidden cell state.
        """
        combined_update_reset = torch.cat([input_tensor, cur_state], dim=1)
        combined_update_reset = torch.sigmoid(self.update_reset_conv(combined_update_reset))
        r, z = torch.split(combined_update_reset, self.hidden_dim, dim=1)

        combined_update = torch.cat([input_tensor, (cur_state * r)], dim=1)
        combined_update = self.update_conv(combined_update)
        h_bar = torch.tanh(combined_update)

        h_next = (1 - z) * cur_state + z * h_bar

        return h_next

    def init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden state.

        Args:
            batch_size: The batch size.
            image_size: The height and width of the image.

        Returns:
            A tensor for the initial hidden state.
        """
        height, width = image_size
        device = device = self.update_reset_conv.weight.device
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)


class ConvGRU(nn.Module):
    """Convolutional GRU model.

    This model is a sequence-processing model that uses convolutional operations
    within the GRU cells. It is particularly useful for spatio-temporal data.

    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | list[int],
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]],
        num_layers: int,
        batch_first: bool = True,
        bias: bool = True,
        return_all_layers: bool = False,
    ) -> None:
        """Initializes the ConvGRU model.

        Args:
            input_dim: Number of channels in the input.
            hidden_dim: Number of hidden channels. Can be a single int (for all
                layers) or a list of ints (one for each layer).
            kernel_size: Size of the convolutional kernel. Can be:

                * a single integer (for square kernels)
                * a tuple of two integers (for rectangular kernels)
                * a list of integers or tuples (one for each layer)
            num_layers: Number of LSTM layers stacked on each other.
            batch_first: If ``True``, then the input and output tensors are
                provided as (b, t, c, h, w).
            bias: If ``True``, adds a learnable bias to the output.
            return_all_layers: If ``True``, will return the list of computations
                for all layers.
        """
        super().__init__()

        # Normalize hidden_dim to a list of ints
        if isinstance(hidden_dim, int):
            self.hidden_dim = [hidden_dim] * num_layers
        else:
            self.hidden_dim = hidden_dim

        # Normalize kernel_size to a list of tuples
        if isinstance(kernel_size, int | tuple):
            ks_list = [kernel_size] * num_layers
        else:
            ks_list = kernel_size

        self.kernel_size = [(ks, ks) if isinstance(ks, int) else ks for ks in ks_list]

        if not len(self.kernel_size) == len(self.hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(
                ConvGRUCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dim[i],
                    kernel_size=self.kernel_size[i],
                    bias=self.bias,
                )
            )

        self.cell_list = nn.ModuleList(cell_list)


    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass of the ConvGRU.

        Args:
            input_tensor: A 5-D Tensor of shape (t, b, c, h, w) or (b, t, c, h, w).
            hidden_state: An optional initial hidden state.

        Returns:
            A tuple containing layer_output_list and last_state_list.
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4).contiguous()

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h_state = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h_state = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=h_state,
                )
                output_inner.append(h_state)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append(h_state)

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list



    def _init_hidden(
        self, batch_size: int, image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        """Initializes the hidden states for all layers.

        Args:
            batch_size: The size of the batch dimension.
            image_size: A tuple of (height, width) for the spatial dimensions.

        Returns:
            A list of tuples, where each tuple contains the hidden state and cell state
            tensors for a layer. Each tensor has shape (batch_size, hidden_dim, height, width).
        """
        init_states = []
        for i in range(self.num_layers):
            cell = cast(ConvGRUCell, self.cell_list[i])
            init_states.append(cell.init_hidden(batch_size, image_size))
        return init_states


class baseline_ConvGRU(nn.Module):
    def __init__(self,
                input_dims: int, 
                hidden_dim: int | list[int],
                kernel_size: int | tuple[int, int] | list[int | tuple[int, int]],
                num_layers: int,
                num_pred_steps: int,
                batch_first: bool = True):
        super().__init__()
        self.encoder = ConvGRU(input_dims,
                            hidden_dim,
                            kernel_size,
                            num_layers,
                            batch_first=batch_first,
                            return_all_layers=True)
        self.decoder = ConvGRU(1,
                            hidden_dim,
                            kernel_size,
                            num_layers,
                            batch_first=batch_first,
                            return_all_layers=True)
        total_hidden = sum(hidden_dim) if isinstance(hidden_dim, list) else hidden_dim * num_layers
        self.forecaster = nn.Conv2d(
            in_channels=total_hidden,
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
        # converts the hidden states into predictions for the future timesteps
        uncompressed_predictions = torch.cat(decoder_states, dim=1)
        output = self.forecaster(uncompressed_predictions)
        return torch.clamp(output, min=0)
