"""Causal residual TCN, GRU and causal Transformer for framewise phases."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalConv(nn.Module):
    def __init__(self, width, kernel=3, dilation=1, depthwise=False):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(width, width, kernel, dilation=dilation, groups=width if depthwise else 1)
        self.pointwise = nn.Conv1d(width, width, 1) if depthwise else nn.Identity()

    def forward(self, x):
        return self.pointwise(self.conv(F.pad(x, (self.pad, 0))))


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation, depthwise, dropout):
        super().__init__()
        self.net = nn.Sequential(CausalConv(width, dilation=dilation, depthwise=depthwise), nn.GELU(),
                                 nn.Dropout(dropout), CausalConv(width, dilation=dilation, depthwise=depthwise),
                                 nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(x)


class PhaseModel(nn.Module):
    def __init__(self, input_dim, kind="tcn", width=32, blocks=3, depthwise=False, dropout=.1, num_classes=17):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(input_dim, width)
        if kind == "tcn":
            self.temporal = nn.Sequential(*[ResidualBlock(width, 2 ** i, depthwise, dropout) for i in range(blocks)])
        elif kind == "gru":
            self.temporal = nn.GRU(width, width, blocks, batch_first=True, dropout=dropout if blocks > 1 else 0)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=width * 2,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.temporal = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        else:
            raise ValueError(kind)
        self.head = nn.Linear(width, num_classes)

    def forward(self, x):
        x = self.project(x)
        if self.kind == "tcn":
            x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        elif self.kind == "gru":
            x = self.temporal(x)[0]
        else:
            t = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)
            freqs = torch.exp(torch.arange(0, x.shape[2], 2, device=x.device, dtype=x.dtype) * (-9.21034 / x.shape[2]))
            position = torch.zeros_like(x[0])
            position[:, 0::2] = torch.sin(t[:, None] * freqs)
            position[:, 1::2] = torch.cos(t[:, None] * freqs)
            mask = torch.triu(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device), diagonal=1)
            x = self.temporal(x + position, mask=mask)
        return self.head(x)


def build_model(config, input_dim):
    if config.get("hierarchical"):
        return HierarchicalPhaseModel(input_dim, **{k: config[k] for k in ["kind", "width", "blocks", "dropout"] if k in config})
    return PhaseModel(input_dim, **{k: config[k] for k in ["kind", "width", "blocks", "depthwise", "dropout"] if k in config})


class NormalizedResidualBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.net = nn.Sequential(CausalConv(width, dilation=dilation), nn.GELU(), nn.Dropout(dropout),
                                 CausalConv(width, dilation=dilation), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.net(self.norm(x).transpose(1, 2)).transpose(1, 2)


class HierarchicalPhaseModel(nn.Module):
    """Independent action and conditional phase heads with shared causal context."""
    def __init__(self, input_dim, kind="tcn", width=48, blocks=4, dropout=.15):
        super().__init__()
        self.kind = kind
        self.project = nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU())
        if kind == "tcn":
            self.temporal = nn.Sequential(*[NormalizedResidualBlock(width, 2 ** i, dropout) for i in range(blocks)])
        elif kind == "gru":
            self.temporal = nn.GRU(width, width, blocks, batch_first=True, dropout=dropout if blocks > 1 else 0)
        else:
            raise ValueError(kind)
        self.norm = nn.LayerNorm(width)
        self.action = nn.Linear(width, 5)
        self.phase = nn.Linear(width + input_dim, 16)

    def components(self, x):
        h = self.project(x)
        h = self.temporal(h)[0] if self.kind == "gru" else self.temporal(h)
        h = self.norm(h)
        return self.action(h), self.phase(torch.cat([h, x], -1)).reshape(*x.shape[:-1], 4, 4)

    def forward(self, x):
        action, phase = self.components(x)
        action = action.log_softmax(-1)
        joint = action[..., :4, None] + phase.log_softmax(-1)
        return torch.cat([joint.flatten(-2), action[..., 4:]], -1)
