"""Stochastic Multi-View Weight Standardization (SMWS).

Multi-View Weight Standardization constructs I-WS, S-WS, and O-WS views.
Stochastic Multi-View Mixing combines these views during training;
inference uses equal coefficients. Weight axes are (output, input, height, width).
"""

import torch
import torch.nn as nn

from model.smws_config import PROJECTION_POLICY


EPSILON = 1e-5
VIEW_DIMS = {"I-WS": (1, 2, 3), "S-WS": (2, 3), "O-WS": (0, 2, 3)}


def sample_dirichlet(reference, sample_shape=torch.Size()):
    """Native PyTorch Dirichlet(1,1,1), controlled by the torch RNG."""
    sample_shape = torch.Size(sample_shape)
    concentration = reference.new_ones(3)
    return torch.distributions.Dirichlet(concentration).sample(sample_shape)


def sample_uniform(reference, sample_shape=torch.Size()):
    """Native PyTorch U(0,1), controlled by the torch RNG."""
    return torch.rand(torch.Size(sample_shape), device=reference.device, dtype=reference.dtype)


def eval_dirichlet(reference):
    return reference.new_full((3,), 1.0 / 3.0)


def i_ws(weight, eps=EPSILON):
    """I-WS: input-spatial statistics for each output channel."""
    var, mean = torch.var_mean(
        weight, dim=(1, 2, 3), keepdim=True, unbiased=False
    )
    return (weight - mean) * torch.rsqrt(var + eps)


def s_ws(weight, eps=EPSILON):
    """S-WS: spatial statistics for each input-output channel pair."""
    var, mean = torch.var_mean(
        weight, dim=(2, 3), keepdim=True, unbiased=False
    )
    return (weight - mean) * torch.rsqrt(var + eps)


def o_ws(weight, eps=EPSILON):
    """O-WS: output-spatial statistics for each input channel."""
    # O-WS fixes input channel i and computes statistics over (o, u, v).
    var, mean = torch.var_mean(
        weight, dim=(0, 2, 3), keepdim=True, unbiased=False
    )
    return (weight - mean) * torch.rsqrt(var + eps)


class StochasticMultiViewMixing:
    """Share coefficients for stochastic multi-view mixing within a stage."""

    def __init__(self):
        self._mix = None

    def begin(self, reference, training):
        self._mix = sample_dirichlet(reference) if training else eval_dirichlet(reference)
        return self._mix

    def current(self, reference):
        if self._mix is None:
            raise RuntimeError("stochastic multi-view mixing coefficients were not initialized")
        if self._mix.device != reference.device or self._mix.dtype != reference.dtype:
            raise RuntimeError("stochastic multi-view mixing coefficients changed device or dtype")
        return self._mix


class SMWSConv2d(nn.Conv2d):
    """SMWS: combine the I-WS, S-WS, and O-WS views of convolutional weights."""

    def __init__(self, *args, mix_controller=None, eps=EPSILON, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kernel_size == (1, 1):
            raise ValueError("SMWS convolutions exclude 1x1 weights because S-WS degenerates")
        if self.groups != 1:
            raise ValueError("SMWS is configured for ResNet-34 convolutions with groups=1")
        self.mix_controller = mix_controller
        self.eps = float(eps)
        self._last_lambda = None

    def _mix_vector(self, weight):
        if self.mix_controller is not None:
            return self.mix_controller.current(weight)
        return sample_dirichlet(weight) if self.training else eval_dirichlet(weight)

    def weight_components(self):
        weight = self.weight
        weight_i = i_ws(weight, self.eps)
        weight_s = s_ws(weight, self.eps)
        weight_o = o_ws(weight, self.eps)
        mix = self._mix_vector(weight)
        self._last_lambda = mix.detach().clone()
        mixed = mix[0] * weight_i + mix[1] * weight_s + mix[2] * weight_o
        return weight_i, weight_s, weight_o, mix, mixed

    def forward(self, input):
        _, _, _, _, effective_weight = self.weight_components()
        return self._conv_forward(input, effective_weight, self.bias)


class ProjectionConv2d(nn.Conv2d):
    """Residual 1x1 projections; the final SMWS setting uses deterministic O-WS."""

    def __init__(self, *args, eps=EPSILON, **kwargs):
        super().__init__(*args, **kwargs)
        if self.kernel_size != (1, 1) or self.groups != 1:
            raise ValueError("projection policy is restricted to groups=1 residual 1x1")
        if PROJECTION_POLICY not in {"RAW", "I-WS", "O-WS", "I-O-Mix"}:
            raise ValueError(f"unsupported projection policy: {PROJECTION_POLICY}")
        self.eps = float(eps)
        self._last_mu = None

    def effective_weight(self):
        weight = self.weight
        if PROJECTION_POLICY == "RAW":
            self._last_mu = None
            return weight
        if PROJECTION_POLICY == "I-WS":
            self._last_mu = None
            return i_ws(weight, self.eps)
        if PROJECTION_POLICY == "O-WS":
            self._last_mu = None
            return o_ws(weight, self.eps)
        mu = sample_uniform(weight) if self.training else weight.new_tensor(0.5)
        self._last_mu = mu.detach().clone()
        return (1.0 - mu) * i_ws(weight, self.eps) + mu * o_ws(weight, self.eps)

    def forward(self, input):
        return self._conv_forward(input, self.effective_weight(), self.bias)
