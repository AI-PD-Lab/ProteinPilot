from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        """Initialize the Exponential Moving Average (EMA) module.

        EMA maintains a shadow model that slowly tracks the weight of the original model.

        Args:
            model: The original model.
            decay: The decay rate of the EMA. The shadow model will be updated with the formula:
                shadow_variable -= (1 - decay) * (shadow_variable - variable)
        """
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.detach_()

    @torch.no_grad()
    def update(self):
        """Update the shadow model using the weight of the original model and the decay rate."""
        if not self.training:
            raise RuntimeError("EMA update should only be called during training")

        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())

        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            if param.requires_grad:
                shadow_params[name].sub_(
                    (1.0 - self.decay) * (shadow_params[name] - param)
                )

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())

        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def forward(self, *args, **kwargs):
        """Dynamic dispatch to the correct model (model or shadow)."""
        if self.training:
            return self.model(*args, **kwargs)
        else:
            return self.shadow(*args, **kwargs)
