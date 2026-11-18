import math

import torch
import torch.nn as nn


class SSRatioEncoder(nn.Module):
    """Encodes a global ss_ratio scalar into a token-level conditioning vector.

    Pipeline:
        scalar [0,1] → fixed Fourier basis (num_freqs sin/cos pairs)
                     → Linear(2*num_freqs, c_s) → SiLU
                     → Linear(c_s, c_s)  ← zero-initialised (safe warm start)

    Sentinel handling:
        ss_ratio = -1.0 means unconditional. The output is zeroed for those
        tokens so the backbone sees no signal, enabling CFG at inference.
    """

    def __init__(self, num_freqs: int, c_s: int):
        super().__init__()
        self.num_freqs = num_freqs

        # Fixed (non-learned) Fourier frequencies: k * π for k = 1 … num_freqs
        freqs = torch.arange(1, num_freqs + 1).float() * math.pi
        self.register_buffer("freqs", freqs)

        self.linear1 = nn.Linear(2 * num_freqs, c_s)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(c_s, c_s)

        # Zero-init: at the start of fine-tuning the encoder contributes
        # nothing to S_I, preserving frozen-checkpoint behaviour exactly.
        nn.init.zeros_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, ss_ratio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ss_ratio: [I, 1]  token-level scalar; -1.0 = unconditional sentinel
        Returns:
            [I, c_s]  additive contribution to S_I (zero for unconditional tokens)
        """
        # Build unconditional mask before clamping destroys the sentinel
        mask = (ss_ratio > -0.5).float()          # [I, 1]  1 = conditioned

        x = ss_ratio.clamp(0.0, 1.0)              # [I, 1]

        # Fixed Fourier expansion:  [I, 1] × [num_freqs] → [I, num_freqs]
        angles = x * self.freqs                    # broadcast: [I, num_freqs]
        fourier = torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )                                          # [I, 2*num_freqs]

        out = self.linear2(self.act(self.linear1(fourier)))  # [I, c_s]

        return out * mask                          # zero out unconditional samples
