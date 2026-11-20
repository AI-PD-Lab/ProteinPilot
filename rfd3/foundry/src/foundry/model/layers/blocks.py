import torch
import torch.nn as nn

pi = torch.acos(torch.zeros(1)).item() * 2


class FourierEmbedding(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.register_buffer("w", torch.zeros(c, dtype=torch.float32))
        self.register_buffer("b", torch.zeros(c, dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.w)
        nn.init.normal_(self.b)

    def forward(
        self,
        t,
    ):
        return torch.cos(2 * pi * (t[..., None] * self.w + self.b))


class Dropout(nn.Module):
    def __init__(self, broadcast_dim=None, p_drop=0.15):
        super(Dropout, self).__init__()
        self.sampler = torch.distributions.bernoulli.Bernoulli(
            torch.tensor([1 - p_drop])
        )
        self.broadcast_dim = broadcast_dim
        self.p_drop = p_drop

    def forward(self, x):
        if not self.training:
            return x
        shape = list(x.shape)
        if self.broadcast_dim is not None:
            shape[self.broadcast_dim] = 1
        mask = self.sampler.sample(shape).to(x.device).view(shape)

        x = mask * x / (1.0 - self.p_drop)
        return x
