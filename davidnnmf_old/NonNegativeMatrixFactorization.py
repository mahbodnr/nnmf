from abc import abstractmethod
from typing import Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from utils import PowerSoftmax

COMPARISSON_TOLERANCE = 1e-5


class NonNegativeParameter(torch.nn.Parameter):
    """
    A parameter that is constrained to be non-negative.
    """

    def __new__(cls, data):
        if torch.any(data < 0):
            raise ValueError(
                "Negative values are not allowed in the parameter data.",
                data[torch.where(data < 0)],
            )
        return super(NonNegativeParameter, cls).__new__(cls, data, requires_grad=True)

    def _check_negative_values(self):
        if torch.any(self.data < 0):
            raise ValueError("Negative values are not allowed in the parameter data.")

    def __setattr__(self, name, value):
        if name == "data":
            self._check_negative_values()
            super(NonNegativeParameter, self).__setattr__(name, value)
        else:
            super(NonNegativeParameter, self).__setattr__(name, value)

    def __setitem__(self, key, value):
        self._check_negative_values()
        super(NonNegativeParameter, self).__setitem__(key, value)

    def __delitem__(self, key):
        self._check_negative_values()
        super(NonNegativeParameter, self).__delitem__(key)

    def __setstate__(self, state):
        self._check_negative_values()
        super(NonNegativeParameter, self).__setstate__(state)


class ParameterList(torch.nn.ParameterList):
    def requires_grad_(self, requires_grad=True):
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def size(self, idx=None):
        if idx is None:
            return [p.size() for p in self.parameters()]
        return self.parameters[idx].size()


class NNMFLayer(nn.Module):
    def __init__(
        self,
        n_iterations,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
        solver=None,
    ):
        super().__init__()
        assert n_iterations >= 0 and isinstance(
            n_iterations, int
        ), f"n_iterations must be a positive integer, got {n_iterations}"
        assert (
            0 < h_update_rate <= 1
        ), f"h_update_rate must be in (0,1], got {h_update_rate}"
        assert backward_method in [
            "fixed_point",
            "solver",
            "all_grads",
        ], f"backward_method must be one of 'fixed_point', 'solver', 'all_grads', got {backward_method}"

        self.n_iterations = n_iterations
        self.activate_secure_tensors = activate_secure_tensors
        self.h_update_rate = h_update_rate
        self.keep_h = keep_h
        self.sparsity = PowerSoftmax(sparsity_rate, dim=1)

        self.backward = backward_method
        if self.backward == "solver":
            assert solver is not None, "solver must be provided when using solver"
            self.solver = solver
            self.hook = None

        self.h = None
        self.normalize_dim = None

    def secure_tensor(self, t):
        if not self.activate_secure_tensors:
            return t
        assert self.normalize_dim is not None, "normalize_dim must be set"
        return F.normalize(F.relu(t), p=1, dim=self.normalize_dim, eps=1e-20)

    @abstractmethod
    def normalize_weights(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_h(self, x):
        raise NotImplementedError

    @abstractmethod
    def _reconstruct(self, h):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, nnmf_update):
        raise NotImplementedError

    @abstractmethod
    def _process_h(self, h):
        raise NotImplementedError

    @abstractmethod
    def _check_forward(self, input):
        """
        Check that the forward pass is valid
        """

    def _nnmf_iteration(self, input):
        X_r = self._reconstruct(self.h)
        # X_r = self.secure_tensor(X_r)
        X_r = F.normalize(X_r.clamp_min(0.0001), p=1, dim=self.normalize_dim, eps=1e-20)
        nnmf_update = input / (X_r + 1e-12)
        # nnmf_update = input / torch.clamp(X_r, min=0.001)
        new_h = self.h * self._forward(nnmf_update)
        h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h)

    def forward(self, input):
        assert self.normalize_dim is not None, "normalize_dim must be set"

        self.normalize_weights()
        self._check_forward(input)
        input = F.normalize(input, p=1, dim=self.normalize_dim, eps=1e-20)

        if (not self.keep_h) or (self.h is None):
            self._reset_h(input)

        if self.backward == "all_grads":
            for _ in range(self.n_iterations):
                self.h = self._nnmf_iteration(input)

        elif self.backward == "fixed_point" or self.backward == "solver":
            with torch.no_grad():
                for _ in range(self.n_iterations - 1):
                    self.h = self._nnmf_iteration(input)

            if self.training:
                if self.backward == "solver":
                    self.h = self.h.requires_grad_()
                new_h = self._nnmf_iteration(input)
                if self.backward == "solver":

                    def backward_hook(grad):
                        if self.hook is not None:
                            self.hook.remove()
                            torch.cuda.synchronize()
                        g, self.backward_res = self.solver(
                            lambda y: torch.autograd.grad(
                                new_h, self.h, y, retain_graph=True
                            )[0]
                            + grad,
                            torch.zeros_like(grad),
                        )
                        return g

                    self.hook = new_h.register_hook(backward_hook)
                self.h = new_h
        return self.h


class NNMFDense(NNMFLayer):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_iterations = n_iterations

        self.weight = NonNegativeParameter(torch.rand(out_features, in_features))
        self.normalize_dim = 1
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=1)

    def _reset_h(self, x):
        self.h = F.normalize(torch.ones(x.shape[0], self.out_features), p=1, dim=1).to(
            x.device
        )

    def _reconstruct(self, h):
        return F.linear(h, self.weight.t())

    def _forward(self, nnmf_update):
        return F.linear(nnmf_update, self.weight)

    def _process_h(self, h):
        # h = self.secure_tensor(h)
        # apply sparsity
        h = self.sparsity(F.relu(h))
        return h

    def _check_forward(self, input):
        assert self.weight.sum(0, keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum(0)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=1, dim=0)


class NNMFConv2d(NNMFLayer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_iterations,
        padding=0,
        stride=1,
        dilation=1,
        normalize_channels=False,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.n_iterations = n_iterations
        self.normalize_channels = normalize_channels
        self.normalize_dim = (1, 2, 3)
        if self.dilation != (1, 1):
            raise NotImplementedError(
                "Dilation not implemented for NNMFConv2d, got dilation={self.dilation}"
            )

        self.weight = NonNegativeParameter(
            torch.rand(out_channels, in_channels, kernel_size, kernel_size)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))

    def normalize_weights(self):
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))

    def _reconstruct(self, h):
        return F.conv_transpose2d(
            h,
            self.weight,
            padding=self.padding,
            stride=self.stride,
        )

    def _forward(self, nnmf_update):
        return F.conv2d(
            nnmf_update, self.weight, padding=self.padding, stride=self.stride
        )

    def _process_h(self, h):
        if self.normalize_channels:
            h = F.normalize(F.relu(h), p=1, dim=1)
        else:
            h = self.secure_tensor(h)
        return h

    def _reset_h(self, x):
        output_size = [
            (x.shape[-2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]
            + 1,
            (x.shape[-1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]
            + 1,
        ]
        self.h = torch.ones(x.shape[0], self.out_channels, *output_size).to(x.device)

    def _check_forward(self, input):
        assert self.weight.sum((1, 2, 3), keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum((1, 2, 3))
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()


class NNMFVJP(NNMFLayer):
    def _nnmf_iteration(self, input):
        if isinstance(self.h, tuple):
            reconstruct = self._reconstruct(*self.h)
        else:
            reconstruct = self._reconstruct(self.h)
        reconstruct = self.secure_tensor(reconstruct)
        nnmf_update = input / (reconstruct + 1e-20)
        h_update = torch.autograd.functional.vjp(
            self._reconstruct,
            self.h,
            nnmf_update,
            create_graph=True,
        )[1]
        if isinstance(self.h, tuple):
            new_h = tuple(
                self.h_update_rate * h_update[i] * self.h[i]
                + (1 - self.h_update_rate) * self.h[i]
                for i in range(len(self.h))
            )
        else:
            new_h = (
                self.h_update_rate * h_update * self.h
                + (1 - self.h_update_rate) * self.h
            )
        return self._process_h(new_h)


class NNMFDenseVJP(NNMFVJP, NNMFDense):
    """
    NNMFDense with VJP backward method
    """


class NNMFConv2dVJP(NNMFVJP, NNMFConv2d):
    """
    NNMFConv2d with VJP backward method
    """


class NNMFConvTransposed2d(NNMFConv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        n_iterations,
        padding=0,
        output_padding=0,
        stride=1,
        dilation=1,
        normalize_channels=False,
        backward_method="fixed point",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            n_iterations,
            padding,
            stride,
            dilation,
            normalize_channels,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.output_padding = _pair(output_padding)
        assert (
            (self.output_padding[0] < self.stride[0])
            or (self.output_padding[0] < self.dilation[0])
        ) and (
            (self.output_padding[1] < self.stride[1])
            or (self.output_padding[1] < self.dilation[1])
        ), f"RuntimeError: output padding must be smaller than either stride or dilation, but got output_padding={self.output_padding}, stride={self.stride}, dilation={self.dilation}"

        self.weight = NonNegativeParameter(
            torch.rand(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.reset_parameters()

    def _reset_h(self, x):
        output_size = [
            (x.shape[-2] - 1) * self.stride[0]
            - 2 * self.padding[0]
            + self.kernel_size[0]
            + self.output_padding[0],
            (x.shape[-1] - 1) * self.stride[1]
            - 2 * self.padding[1]
            + self.kernel_size[1]
            + self.output_padding[1],
        ]
        self.h = torch.ones(x.shape[0], self.out_channels, *output_size).to(x.device)

    def _nnmf_iteration(self, input, h):
        X_r = F.conv2d(h, self.weight, padding=self.padding, stride=self.stride)
        X_r = self.secure_tensor(X_r, dim=(1, 2, 3))
        if X_r.shape != input.shape:
            input = F.pad(
                input,
                [
                    0,
                    X_r.shape[-1] - input.shape[-1],
                    0,
                    X_r.shape[-2] - input.shape[-2],
                ],
            )
        nnmf_update = input / (X_r + 1e-20)
        new_h = h * F.conv_transpose2d(
            nnmf_update,
            self.weight,
            padding=self.padding,
            stride=self.stride,
            output_padding=self.output_padding,
        )
        h = self.h_update_rate * new_h + (1 - self.h_update_rate) * h
        if self.normalize_channels:
            # h = F.normalize(F.relu(h), p=1, dim=1)
            h = self.sparsity(F.relu(h))
        else:
            h = self.secure_tensor(h, dim=(1, 2, 3))
        return h


class NNMFMultiscale(NNMFVJP):
    def __init__(
        self,
        kernel_sizes: List[int],
        in_channels: int,
        out_channels: Union[int, List[int]],
        n_iterations: Union[int, List[int]],
        paddings: Union[int, List[int]] = 0,
        strides: Union[int, List[int]] = 1,
        dilations: Union[int, List[int]] = 1,
        normalize_channels=False,
        backward_method="fixed point",
        h_update_rate=1,
        sparsity_rate=1,
        keep_h=False,
        activate_secure_tensors=False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        if not isinstance(kernel_sizes, tuple):
            kernel_sizes = list(kernel_sizes)
        assert isinstance(
            kernel_sizes, list
        ), f"kernel_sizes must be a list, got {kernel_sizes}"
        if len(kernel_sizes) == 1:
            print(
                "Warning: a Multiscale module with just one kernel. You should use NNMFConv2d instead."
            )
        self.kernel_sizes = kernel_sizes
        self.n_scales = len(self.kernel_sizes)
        self.in_channels = in_channels
        self.out_channels = (
            out_channels
            if isinstance(out_channels, list)
            else [out_channels] * self.n_scales
        )
        self.stride = (
            strides if isinstance(strides, list) else [strides] * self.n_scales
        )
        self.padding = (
            paddings if isinstance(paddings, list) else [paddings] * self.n_scales
        )
        self.dilation = (
            dilations if isinstance(dilations, list) else [dilations] * self.n_scales
        )
        self.n_iterations = n_iterations
        self.normalize_channels = normalize_channels
        assert (
            len(self.out_channels)
            == len(self.stride)
            == len(self.padding)
            == len(self.dilation)
            == len(self.kernel_sizes)
        ), f"Number of kernels must be the same for all parameters, got {self.out_channels}, {self.stride}, {self.padding}, {self.dilation}, {self.kernel_sizes}"
        for dilation in self.dilation:
            if dilation != 1:
                raise NotImplementedError(
                    "Dilation > 1 not implemented for NNMFMultiscale, got dilation={self.dilation}"
                )

        self.normalize_dim = (1, 2, 3)
        self.weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.rand(out_channel, in_channels, kernel_size, kernel_size)
                )
                for out_channel, kernel_size in zip(
                    self.out_channels, self.kernel_sizes
                )
            ]
        )

    def reset_parameters(self):
        for weight in self.weights:
            nn.init.uniform_(weight, a=0, b=1)
            weight.data = F.normalize(weight.data, p=1, dim=(1, 2, 3))

    def normalize_weights(self):
        for weight in self.weights:
            weight.data = F.normalize(weight.data, p=1, dim=(1, 2, 3))

    def _reconstruct(self, *h_list):
        regenerated = []
        for weight, padding, stride, h in zip(
            self.weights, self.padding, self.stride, h_list
        ):
            X_r = F.conv_transpose2d(
                h,
                weight,
                padding=padding,
                stride=stride,
                output_padding=(
                    self.input_shape[-2]
                    - ((h.shape[-2] - 1) * stride - 2 * padding + weight.shape[-2]),
                    self.input_shape[-1]
                    - ((h.shape[-1] - 1) * stride - 2 * padding + weight.shape[-1]),
                ),
            )
            regenerated.append(X_r)

        return torch.stack(regenerated, dim=0).sum(0)

    def _reset_h(self, x):
        self.h = []
        self.input_shape = x.shape
        for i in range(self.n_scales):
            output_size = (
                (x.shape[-1] - self.kernel_sizes[i] + 2 * self.padding[i])
                // self.stride[i]
                + 1,
                (x.shape[-2] - self.kernel_sizes[i] + 2 * self.padding[i])
                // self.stride[i]
                + 1,
            )
            h = torch.ones(
                x.shape[0], self.out_channels[i], *output_size, requires_grad=False
            ).to(x.device)
            self.h.append(h)
        self.h = tuple(self.h)

    def _check_forward(self, input):
        for weight in self.weights:
            assert weight.sum((1, 2, 3), keepdim=True).allclose(
                torch.ones_like(weight), atol=COMPARISSON_TOLERANCE
            ), weight.sum((1, 2, 3))
            assert (weight >= 0).all(), weight.min()
        assert (input >= 0).all(), input.min()

    def _process_h(self, h_list):
        new_h_list = []
        if self.normalize_channels:
            for h in h_list:
                new_h_list.append(F.normalize(F.relu(h), p=1, dim=1))
        else:
            for h in h_list:
                new_h_list.append(self.secure_tensor(h))
        return tuple(new_h_list)


"""
class NNMFAttentionHeads(NNMFVJP):
    def __init__(
        self,
        features: int,
        heads: int,
        head_features: int,
        n_iterations: int,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = False,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        self.normalize_dim = 1

        self.features = features
        self.heads = heads
        self.head_features = head_features
        self.n_iterations = n_iterations
        self.backward = backward_method
        self.h_update_rate = h_update_rate
        self.keep_h = keep_h
        # self.sparsity = PowerSoftmax(sparsity_rate, dim=1)

        heads_dim = self.heads * self.head_features
        self.sqrt_d = torch.sqrt(torch.tensor(self.head_features, dtype=torch.float))

        self.key_weight = PositiveParameter(torch.rand(heads_dim, self.features))
        self.query_weight = PositiveParameter(torch.rand(heads_dim, self.features))
        self.value_weight = PositiveParameter(torch.rand(heads_dim, self.features))

        self.out_proj = PositiveParameter(torch.rand(self.features, heads_dim))
        self.save_attn_map = False
        self.reset_parameters()

    def _reset_h(self, x):
        self.h = torch.ones_like(x)

    def _process_h(self, h):
        h = self.secure_tensor(h)
        return h

    def _reconstruct(self, h):
        batch_size, seq_len, dim = h.shape
        query = (
            torch.Linear(h, self.query_weight)
            .view(batch_size, seq_len, self.heads, self.head_features)
            .transpose(1, 2)
        )
        key = (
            torch.Linear(h, self.key_weight)
            .view(batch_size, seq_len, self.heads, self.head_features)
            .transpose(1, 2)
        )
        value = (
            torch.Linear(h, self.value_weight)
            .view(batch_size, seq_len, self.heads, self.head_features)
            .transpose(1, 2)
        )

        attn_map = F.softmax(
            torch.einsum("bhif,bhjf->bhij", query, key) / self.sqrt_d, dim=-1
        )
        if self.save_attn_map:
            self.attn_map = attn_map
        attn = torch.einsum("bhij,bhjf->bihf", attn_map, value).flatten(2)
        output = torch.Linear(attn, self.out_proj)
        return output

    def reset_parameters(self):
        nn.init.uniform_(self.key_weight, a=0, b=1)
        self.key_weight.data = F.normalize(self.key_weight.data, p=1, dim=0)
        nn.init.uniform_(self.query_weight, a=0, b=1)
        self.query_weight.data = F.normalize(self.query_weight.data, p=1, dim=0)
        nn.init.uniform_(self.value_weight, a=0, b=1)
        self.value_weight.data = F.normalize(self.value_weight.data, p=1, dim=0)
        nn.init.uniform_(self.out_proj, a=0, b=1)
        self.out_proj.data = F.normalize(self.out_proj.data, p=1, dim=0)

    def _check_forward(self, input):
        assert (self.key_weight >= 0).all(), self.key_weight.min()
        assert (self.query_weight >= 0).all(), self.query_weight.min()
        assert (self.value_weight >= 0).all(), self.value_weight.min()
        assert (self.out_proj >= 0).all(), self.out_proj.min()
        assert (input >= 0).all(), input.min()

    def normalize_weights(self):
        self.key_weight.data = F.normalize(self.key_weight.data, p=1, dim=0)
        self.query_weight.data = F.normalize(self.query_weight.data, p=1, dim=0)
        self.value_weight.data = F.normalize(self.value_weight.data, p=1, dim=0)
        self.out_proj.data = F.normalize(self.out_proj.data, p=1, dim=0)
"""


# """
class NNMFAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
        output_caps=None,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        threshold: float = 0.00001
        self.threshold: float = threshold
        self.heads: int = heads

        self.input_dim: int = features // heads
        self.input_caps: int = seq_len
        self.output_dim: int = features // heads
        self.output_caps: int = seq_len if output_caps is None else output_caps
        self.threshold: float = threshold
        self.weight: NonNegativeParameter = NonNegativeParameter(
            torch.ones(
                self.input_caps,
                self.output_caps,
                self.input_dim,
                self.output_dim,
            )
        )

        self.sqrt_d = features**0.5
        self.normalize_dim = -1
        # self.U = NonNegativeParameter(torch.rand(features, features))
        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)
        self.in_project = nn.Linear(features, features)
        self.save_attn_map = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        weight_data = F.normalize(
            self.weight.data, p=1, dim=(0, 2)
        )  # May contain negative values if Madam not used
        weight_data = torch.clamp(
            weight_data,
            min=self.threshold,
            max=None,
            out=self.weight.data,
        )
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(0, 2))

        # U_weight_data = F.normalize(
        #     self.U.data, p=1, dim=1
        # )  # May contain negative values if Madam not used
        # U_weight_data = torch.clamp(
        #     U_weight_data,
        #     min=self.threshold,
        #     max=None,
        #     out=self.U.data,
        # )
        # self.U.data = F.normalize(self.U.data, p=1, dim=1)

    def _reset_h(self, x):
        self.h = torch.full(
            (
                x.shape[0],
                self.heads,
                self.input_caps,
                self.output_caps,
                self.output_dim,
            ),
            1.0 / float(self.output_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def _reconstruct(self, h):
        return torch.einsum(
            "bhiof,iodf->bhiod", h, self.weight
        )  # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _forward(self, nnmf_update):
        return torch.einsum(
            "bhiod,iodf->bhiof", nnmf_update, self.weight
        )  # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def _check_forward(self, input):
        # assert self.weight.sum(1, keepdim=True).allclose(
        #     torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        # ), self.weight.sum(1)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def routing(self, x):
        h_w = torch.einsum("bhiof,iodf->bhiod", self.h, self.weight)
        alpha = torch.einsum("bhid,bhiod->bhoi", x, h_w)
        # alpha = F.softmax(alpha / self.sqrt_d, dim=-1)
        alpha = F.normalize(alpha, p=1, dim=-1)
        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_project(x)
        x = torch.clamp(x, min=0.0001)
        B, T, _ = x.size()  # (#Batches, #Inputs, #Features)
        x = x.reshape(B, T, self.heads, self.input_dim).permute(
            0, 2, 1, 3
        )  # B, H, T, D
        super().forward(x.unsqueeze(-2))  # B, H, T, 1, D
        alpha = self.routing(x)
        if self.save_attn_map:
            self.attn_map = alpha
        attn = torch.einsum("bhoi,bhid->bohd", alpha, x)
        # attn = torch.einsum("bhio,bhid->bohd", alpha, x)
        output = attn.flatten(2)
        # TODO:
        # output = F.normalize(output, p=1, dim=-1)
        if self.use_out_proj:
            output = self.out_project(output)
        return output

    def norm_weights(self) -> None:
        self.normalize_weights()


# """

"""

class NNMFAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        heads: int,
        head_features: int,
        n_iterations: int,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        threshold: float = 0.00001
        self.threshold: float = threshold
        self.heads: int = heads

        self.input_dim: int = features // heads
        self.input_caps: int = seq_len * heads
        self.output_dim: int = features // heads
        self.output_caps: int = seq_len * heads
        self.threshold: float = threshold
        self.weight: NonNegativeParameter = NonNegativeParameter(
            torch.ones(
                self.input_caps,
                self.output_caps ,
                self.input_dim ,
                self.output_dim,
            )
        )

        self.normalize_dim = -1
        self.reset_parameters()
        self.out_project = nn.Linear(features, features)
        self.save_attn_map = False

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        weight_data = F.normalize(
            self.weight.data, p=1, dim=(1,3)
        )  # May contain negative values if Madam not used

        weight_data = torch.clamp(
            weight_data,
            min=self.threshold,
            max=None,
            out=self.weight.data,
        )

        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1,3))

    def _reset_h(self, x):
        self.h = torch.full(
            (x.shape[0], self.input_caps, self.output_caps, self.output_dim),
            1.0 / float(self.output_dim),
            dtype=x.dtype,
            device=x.device,
        )

    def _reconstruct(self, h):
        return torch.einsum(
            "biof,iodf->bid", h, self.weight
        ) # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _forward(self, nnmf_update):
        return torch.einsum(
            "bid,iodf->biof", nnmf_update, self.weight
        )  # b: batch, i: input_caps, o: output_caps, d: input_dim, f: output_dim

    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def _check_forward(self, input):
        # assert self.weight.sum(1, keepdim=True).allclose(
        #     torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        # ), self.weight.sum(1)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def routing(self, x):
        h_w = torch.einsum("biof,iodf->biod", self.h, self.weight)
        alpha = torch.einsum("bid,biod->bio", x, h_w)
        # alpha = F.normalize(alpha, p=1, dim=-1)
        return alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        B, T, f = input.size()  # (#Batches, #Inputs, #Features)
        input = input.reshape(B, self.input_caps, self.input_dim)
        super().forward(input)
        alpha = self.routing(input)
        if self.save_attn_map:
            self.attn_map = alpha
        attn = torch.einsum("bio,bid->bod", alpha, input)
        output = attn.reshape(B, T, f)
        output = F.normalize(output, p=1, dim=-1)
        output = self.out_project(output)
        return output

    def norm_weights(self) -> None:
        self.normalize_weights()
"""

"""
class NLCAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
        output_caps=None,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
        )
        threshold: float = 0.00001
        self.threshold: float = threshold
        self.heads: int = heads
        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(features, features)
        )
        self.global_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(seq_len, seq_len)
        )
        self.sqrt_d = features**0.5
        self.normalize_dim = -1
        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)
        self.save_attn_map = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        for weight in [self.local_weight, self.global_weight]:
            weight_data = F.normalize(
                weight.data, p=1, dim=1
            )  # May contain negative values if Madam not used
            torch.clamp(
                weight_data,
                min=self.threshold,
                max=None,
                out=weight.data,
            )
            weight.data = F.normalize(weight.data, p=1, dim=1)


    def _reset_h(self, x):
        self.h = F.normalize(torch.ones_like(x), p=1, dim=-1)

    def _reconstruct(self, h):
        h = torch.einsum("bof,oi->bif", h, self.global_weight)
        return F.linear(h, self.local_weight.t())

    def _forward(self, x):
        x = F.linear(x, self.local_weight)
        return torch.einsum(
            "bif,oi->bof", x, self.global_weight
        ) 
    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.out_project(x)
        return x

    def _check_forward(self, input):
        # assert self.weight.sum(1, keepdim=True).allclose(
        #     torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        # ), self.weight.sum(1)
        assert (self.local_weight >= 0).all(), self.local_weight.min()
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()
"""


class NLCAttentionHeads(NNMFLayer):
    def __init__(
        self,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
        output_caps=None,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        solver=None,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
            solver,
        )
        threshold: float = 0.00001
        self.threshold: float = threshold
        self.heads: int = heads
        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(features // heads, features // heads)
        )
        self.output_caps = output_caps
        if output_caps is None:
            self.global_weight: NonNegativeParameter = NonNegativeParameter(
                torch.rand(seq_len, seq_len)
            )
        else:
            self.global_weight: NonNegativeParameter = NonNegativeParameter(
                torch.rand(output_caps, seq_len)
            )
        self.sqrt_d = features**0.5
        self.normalize_dim = -1
        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)
        self.save_attn_map = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        for weight in [self.local_weight, self.global_weight]:
            weight_data = F.normalize(
                weight.data, p=1, dim=1
            )  # May contain negative values if Madam not used
            torch.clamp(
                weight_data,
                min=self.threshold,
                max=None,
                out=weight.data,
            )
            weight.data = F.normalize(weight.data, p=1, dim=1)

    def _reset_h(self, x):
        if self.output_caps is None:
            self.h = F.normalize(torch.ones_like(x), p=1, dim=-1)
        else:
            self.h = F.normalize(
                torch.ones(x.shape[0], self.output_caps, *x.shape[2:]), p=1, dim=-1
            ).to(x.device)

    def _reconstruct(self, h):
        h = torch.einsum("bohf,oi->bihf", h, self.global_weight)
        return F.linear(h, self.local_weight.t())

    def _forward(self, x):
        x = F.linear(x, self.local_weight)
        return torch.einsum("bihf,oi->bohf", x, self.global_weight)

    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        x = super().forward(x)
        x = x.flatten(-2)
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def _check_forward(self, input):
        # assert self.weight.sum(1, keepdim=True).allclose(
        #     torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        # ), self.weight.sum(1)
        assert (self.local_weight >= 0).all(), self.local_weight.min()
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()


class NLCAttentionHeadsConv(NNMFLayer):
    def __init__(
        self,
        kernel_size: int,
        stride: int,
        padding: int,
        seq_len: int,
        features: int,
        heads: int,
        n_iterations: int,
        output_caps=None,
        use_out_proj: bool = True,
        backward_method: str = "fixed point",
        h_update_rate: float = 1,
        sparsity_rate: float = 1,
        keep_h: bool = False,
        activate_secure_tensors: bool = True,
        solver=None,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            sparsity_rate,
            keep_h,
            activate_secure_tensors,
            solver,
        )
        self.threshold: float = 0.00001
        self.heads: int = heads
        self.features: int = features
        self.seq_len: int = seq_len
        self.local_weight: NonNegativeParameter = NonNegativeParameter(
            torch.rand(features // heads, features // heads)
        )
        self.normalize_dim = -1
        self.use_out_proj = use_out_proj
        if self.use_out_proj:
            self.out_project = nn.Linear(features, features)
        self.save_attn_map = False
        self.patch_size = int(self.seq_len**0.5)
        assert self.patch_size**2 == self.seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.global_weight = nn.Parameter(
            torch.rand(features, self.heads, kernel_size, kernel_size)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform_(self.local_weight, a=0, b=1)
        torch.nn.init.uniform_(self.global_weight, a=0, b=1)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self) -> None:
        assert self.threshold >= 0

        weight_data = F.normalize(
            self.local_weight.data, p=1, dim=1
        )  # May contain negative values if Madam not used
        torch.clamp(
            weight_data,
            min=self.threshold,
            max=None,
            out=self.local_weight.data,
        )
        self.local_weight.data = F.normalize(self.local_weight.data, p=1, dim=1)

        torch.clamp(
            self.global_weight.data,
            min=self.threshold,
            max=None,
            out=self.global_weight.data,
        )
        # weight_data = F.normalize(self.global_weight.data, p=1, dim=(1, 2, 3))
        # torch.clamp(
        #     weight_data,
        #     min=self.threshold,
        #     max=None,
        #     out=self.global_weight.data,
        # )
        # self.global_weight.data = F.normalize(self.global_weight.data, p=1, dim=(1, 2, 3))

    def _make_global_weight(self):
        return F.normalize(
            self.global_weight.repeat_interleave(self.features // self.heads, dim=1),
            p=1,
            dim=(1, 2, 3),
        )  # output_channels, input_channels, kernel_size, kernel_size

    def _reconstruct(self, h):
        # h: B, T, F (=H*D)
        h = h.reshape(h.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        # h = F.conv_transpose2d(
        #     h, self.global_weight, stride=self.stride, padding=self.padding, groups=96
        # )
        h = F.conv_transpose2d(
            h, self.global_weight_conv, stride=self.stride, padding=self.padding
        )
        h = h.flatten(-2).permute(0, 2, 1)  # B, T, HD
        h = h.reshape(h.shape[0], h.shape[1], self.heads, -1)  # B, T, H, D
        return F.linear(h, self.local_weight.t()).flatten(-2)  # B, T, F

    def _forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], self.heads, -1)  # B, T, H, D
        x = F.linear(x, self.local_weight)
        x = x.reshape(x.shape[0], self.patch_size, self.patch_size, -1).permute(
            0, 3, 1, 2
        )  # B, HD, P, P
        # x = F.conv2d(x, self.global_weight, stride=self.stride, padding=self.padding)
        x = F.conv2d(
            x, self.global_weight_conv, stride=self.stride, padding=self.padding
        )
        x = x.flatten(-2).permute(0, 2, 1)  # B, T, HD
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.global_weight_conv = self._make_global_weight()
        x = super().forward(x)
        # x = x.flatten(-2)
        if self.use_out_proj:
            x = self.out_project(x)
        return x

    def _reset_h(self, x):
        self.h = F.normalize(torch.ones_like(x), p=1, dim=-1)

    def _check_forward(self, input):
        assert (self.global_weight >= 0).all(), self.global_weight.min()
        assert (input >= 0).all(), input.min()
        assert (self.local_weight >= 0).all(), self.local_weight.min()

    def _process_h(self, h):
        h = F.normalize(h, p=1, dim=-1)
        # apply sparsity
        # h = self.sparsity(F.relu(h))
        return h


if __name__ == "__main__":
    from torch.autograd import gradcheck

    # test dense
    dense = NNMFDense(5, 3, n_iterations=1)
    dense_vjp = NNMFDenseVJP(5, 3, n_iterations=1)
    dense_weight = F.normalize(torch.rand(3, 5), p=1, dim=0)
    dense.weight.data = dense_weight
    dense_vjp.weight.data = dense_weight
    input1 = torch.rand(1, 5).requires_grad_()
    input2 = input1.clone().detach().requires_grad_()
    # test = gradcheck(dense, input1, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    # test_vjp = gradcheck(dense_vjp, input2, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    dense_out = dense(input1)
    dense_vjp_out = dense_vjp(input2)
    print("dense_out == dense_vjp_out:", torch.allclose(dense_out, dense_vjp_out))
    # compare their gradients
    dense_out.sum().backward()
    dense_vjp_out.sum().backward()
    print(
        "dense_out.grad == dense_vjp_out.grad:",
        torch.allclose(input1.grad, input2.grad),
    )

    # test conv
    conv = NNMFConv2d(1, 1, 3, n_iterations=1)
    conv_vjp = NNMFConv2dVJP(1, 1, 3, n_iterations=1)
    conv_weight = F.normalize(torch.rand(1, 1, 3, 3), p=1, dim=(1, 2, 3))
    conv.weight.data = conv_weight
    conv_vjp.weight.data = conv_weight
    input1 = torch.rand(1, 1, 5, 5).requires_grad_()
    input2 = input1.clone().detach().requires_grad_()
    # test = gradcheck(conv, input, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    # test_vjp = gradcheck(conv_vjp, input, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    conv_out = conv(input1)
    conv_vjp_out = conv_vjp(input2)
    print("conv_out == conv_vjp_out:", torch.allclose(conv_out, conv_vjp_out))
    # compare their gradients
    conv_out.sum().backward()
    conv_vjp_out.sum().backward()
    print(
        "conv_out.grad == conv_vjp_out.grad:",
        torch.allclose(input1.grad, input2.grad),
    )
    # test attention heads
    attn = NNMFAttentionHeads(5, 3, 2, n_iterations=1)
    input = torch.rand(2, 10, 5).requires_grad_()
    # test = gradcheck(attn, input, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    # print(test)
    attn_out = attn(input)
    print(attn_out.shape)

    exit()

    # test multiscale
    multiscale = NNMFMultiscale(
        kernel_sizes=[3, 4, 5],
        in_channels=1,
        out_channels=[2, 4, 6],
        paddings=[1, 2, 2],
        strides=[1, 2, 3],
        n_iterations=1,
    )
    input = torch.rand(1, 1, 16, 16).requires_grad_()

    # test conv
    conv = NNMFConv2d(1, 1, 3, n_iterations=1)
    input = torch.rand(1, 1, 5, 5).requires_grad_()
    test = gradcheck(conv, input, eps=1e-5, atol=1e-3, check_undefined_grad=False)
    print(test)
