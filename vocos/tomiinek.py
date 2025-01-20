import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from typing import Tuple

class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        groups: int = 1,
        stride: int = 1,
        pad_mode: str = "constant",
    ):
        super().__init__()
        assert pad_mode in {"constant", "replicate"}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.effective_kernel_size = dilation * (kernel_size - 1) + 1
        self.padding = self.effective_kernel_size - self.stride
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=bias,
            groups=groups,
            stride=stride,
        )

    def forward(self, x, pad_to_ideal: bool = False):
        extra_padding = 0
        if pad_to_ideal:
            n_frames = (x.shape[-1] - self.stride) / self.stride + 1
            ideal_length = math.ceil(n_frames) * self.stride
            extra_padding = ideal_length - x.shape[-1]
        x = F.pad(x, (self.padding, extra_padding, 0, 0), mode=self.pad_mode)
        x = self.conv(x)
        return x

    def init_state(
        self, batch: int, device: torch.device, dtype: torch.dtype = torch.float
    ) -> torch.Tensor:
        if self.pad_mode == "constant":
            return torch.zeros(
                (batch, self.in_channels, self.padding), device=device, dtype=dtype
            )
        else:
            return torch.stack(
                [
                    torch.zeros(
                        (batch, self.in_channels, self.padding),
                        device=device,
                        dtype=dtype,
                    ),
                    torch.ones(
                        (batch, self.in_channels, self.padding),
                        device=device,
                        dtype=dtype,
                    ),
                ],
                dim=1,
            )

    def online_inference(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.padding != 0:
            if self.pad_mode == "constant":
                x = torch.cat((state, x), dim=2)
                state = x[..., -self.padding :]
            elif self.pad_mode == "replicate":
                state[:, 1, ...] *= x[:, :, 0].unsqueeze(-1)
                x = torch.cat((state.sum(dim=1), x), dim=-1)
                state[:, 1, ...] = 0.0
                state[:, 0, ...] = x[..., -self.padding :]
            else:
                raise NotImplementedError()
        y = self.conv(x)
        return y, state