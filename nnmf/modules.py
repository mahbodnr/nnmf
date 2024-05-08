from abc import abstractmethod
from typing import Union, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnmf import DEBUG

if DEBUG:
    import debug.functional as F

from torch.nn.modules.utils import _pair

from nnmf.utils import PowerSoftmax
from nnmf.parameters import NonNegativeParameter
from nnmf.autograd import FunctionalNNMFLinear, FunctionalNNMFConv2d

COMPARISSON_TOLERANCE = 1e-5
SECURE_TENSOR_MIN = 1e-5


class NNMFLayer(nn.Module):
    def __init__(
        self,
        n_iterations,
        backward_method="all_grads",
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        convergence_threshold=0,
        phantom_damping_factor=0.5,
        unrolling_steps=5,
        normalize_input=False,
        normalize_input_dim=None,
        normalize_reconstruction=False,
        normalize_reconstruction_dim=None,
    ):
        super().__init__()
        assert n_iterations >= 0 and isinstance(
            n_iterations, int
        ), f"n_iterations must be a positive integer, got {n_iterations}"
        assert (
            0 < h_update_rate <= 1
        ), f"h_update_rate must be in (0,1], got {h_update_rate}"
        assert backward_method in [
            "last_iter",
            "implicit",
            "all_grads",
            "phantom_unrolling",
            "david",
        ], f"backward_method must be one of 'last_iter', 'implicit', 'all_grads', 'phantom_unrolling', 'david', got {backward_method}"
        if not activate_secure_tensors:
            warnings.warn(
                "[WARNING] 'activate_secure_tensors' is False! This may lead to numerical instability."
            )

        self.n_iterations = n_iterations
        self.activate_secure_tensors = activate_secure_tensors
        self.return_reconstruction = return_reconstruction
        self.h_update_rate = h_update_rate
        self.keep_h = keep_h
        self.convergence_threshold = convergence_threshold

        self.backward_method = backward_method
        self.phantom_damping_factor = phantom_damping_factor
        self.unrolling_steps = unrolling_steps
        self.normalize_input = normalize_input
        self.normalize_input_dim = normalize_input_dim
        if self.normalize_input and self.normalize_input_dim is None:
            warnings.warn(
                "[WARNING] normalize_input is True but normalize_input_dim is None! This will normalize the entire input tensor (including batch dimension)"
            )
        self.normalize_reconstruction = normalize_reconstruction
        self.normalize_reconstruction_dim = normalize_reconstruction_dim
        if self.normalize_reconstruction and self.normalize_reconstruction_dim is None:
            warnings.warn(
                "[WARNING] normalize_reconstruction is True but normalize_reconstruction_dim is None! This will normalize the entire reconstruction tensor (including batch dimension)"
            )
        self.h = None
        self.reconstruction = None
        self.convergence = None
        self.reconstruction_mse = None
        self.forward_iterations = None
        self.prepared_input = None
        if self.backward_method == "implicit":
            self.hook = None
            if return_reconstruction:
                warnings.warn(
                    "[WARNING] return_reconstruction is True but backward_method is 'implicit'! Implicit derivation is not yet implemented for the reconstruction. This will return the reconstruction of the last iteration, not the fixed point."
                )

    def _secure_tensor(self, t):
        return t.clamp_min(SECURE_TENSOR_MIN) if self.activate_secure_tensors else t

    @abstractmethod
    def normalize_weights(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_h(self, x):
        raise NotImplementedError

    @abstractmethod
    def _reconstruct(self, h, input=None, weight=None):
        raise NotImplementedError

    @abstractmethod
    def _forward(self, nnmf_update, weight=None):
        raise NotImplementedError

    @abstractmethod
    def _process_h(self, h):
        raise NotImplementedError

    @abstractmethod
    def _process_reconstruction(self, reconstruction):
        return reconstruction

    @abstractmethod
    def jacobian(self, input, h):
        """
        Compute the jacobian of the forward pass with respect to h* (at the fixed point)

        Can also be computed with torch.autograd.functional.jacobian as:
            jacobian = torch.autograd.functional.jacobian(
                    lambda h: h - h * self._get_nnmf_update(input, h)[0],
                    self.h,
                )
            jacobian = jacobian.sum(<second batch dimension>)

        Torch jacbian returns a tensor of shape (*self.h.shape, *self.h.shape).
        Should me summed over the second batch dimension.
        """
        batch_size, *dims = h.shape
        prod_dims = torch.prod(torch.tensor(dims))

        jacobian = torch.autograd.functional.jacobian(
            lambda h: h - h * self._get_nnmf_update(input, h)[0],
            self.h,
        )
        return jacobian.sum(-(len(dims) + 1)).reshape(batch_size, prod_dims, prod_dims)

    @abstractmethod
    def _check_forward(self, input):
        """
        Check that the forward pass is valid
        """

    def _get_nnmf_update(self, input, h):
        reconstruction = self._reconstruct(h, input=input)
        reconstruction = self._secure_tensor(reconstruction)
        if self.normalize_reconstruction:
            reconstruction = F.normalize(
                reconstruction, p=1, dim=self.normalize_reconstruction_dim, eps=1e-20
            )
        return self._forward(input / reconstruction), reconstruction

    def _nnmf_iteration(self, input):
        nnmf_update, reconstruction = self._get_nnmf_update(input, self.h)
        new_h = self.h * nnmf_update
        if self.h_update_rate == 1:
            h = new_h
        else:
            h = self.h_update_rate * new_h + (1 - self.h_update_rate) * self.h
        return self._process_h(h), self._process_reconstruction(reconstruction)

    def _prepare_input(self, input):
        if self.normalize_input:
            input = F.normalize(input, p=1, dim=self.normalize_input_dim, eps=1e-20)
        return input

    def forward(self, input):
        self.normalize_weights()
        self._check_forward(input)
        input = self._prepare_input(input)

        # save the processed input to be accessed if needed
        self.prepared_input = input

        if (not self.keep_h) or (self.h is None):
            self._reset_h(input)

        self.convergence = []
        self.reconstruction_mse = []
        if self.backward_method == "all_grads":
            for i in range(self.n_iterations):
                self.forward_iterations = i + 1
                new_h, self.reconstruction = self._nnmf_iteration(input)
                self.convergence.append(F.mse_loss(new_h, self.h))
                self.reconstruction_mse.append(F.mse_loss(self.reconstruction, input))
                self.h = new_h
                if (
                    self.convergence_threshold > 0
                    and self.convergence[-1] < self.convergence_threshold
                ):
                    break

        elif self.backward_method == "last_iter":
            with torch.no_grad():
                self.forward_iterations = 0
                no_grad_iterations = (
                    self.n_iterations if not self.training else self.n_iterations - 1
                )
                for i in range(no_grad_iterations):
                    self.forward_iterations = i + 1
                    new_h, self.reconstruction = self._nnmf_iteration(input)
                    self.convergence.append(F.mse_loss(new_h, self.h))
                    self.reconstruction_mse.append(
                        F.mse_loss(self.reconstruction, input)
                    )
                    self.h = new_h
                    if (
                        self.convergence_threshold > 0
                        and self.convergence[-1] < self.convergence_threshold
                    ):
                        break

            if self.training:
                self.forward_iterations += 1
                new_h, self.reconstruction = self._nnmf_iteration(input)
                self.convergence.append(F.mse_loss(new_h, self.h))
                self.reconstruction_mse.append(F.mse_loss(self.reconstruction, input))
                self.h = new_h

        elif self.backward_method == "implicit":
            with torch.no_grad():
                self.forward_iterations = 0
                no_grad_iterations = (
                    self.n_iterations if not self.training else self.n_iterations - 1
                )
                for i in range(no_grad_iterations):
                    self.forward_iterations = i + 1
                    new_h, self.reconstruction = self._nnmf_iteration(input)
                    self.convergence.append(F.mse_loss(new_h, self.h))
                    self.reconstruction_mse.append(
                        F.mse_loss(self.reconstruction, input)
                    )
                    self.h = new_h
                    if (
                        self.convergence_threshold > 0
                        and self.convergence[-1] < self.convergence_threshold
                    ):
                        break

            if self.training:
                self.forward_iterations += 1
                self.h = self.h.requires_grad_()
                self.h, self.reconstruction = self._nnmf_iteration(input)
                jacobian = self.jacobian(input, self.h)

                if self.hook is not None:
                    self.hook.remove()
                    torch.cuda.synchronize()
                self.hook = self.h.register_hook(
                    lambda grad: torch.linalg.solve(
                        A=jacobian.transpose(-1, -2),
                        B=grad.reshape(grad.shape[0], -1),
                    ).reshape(grad.shape)
                )

        elif self.backward_method == "phantom_unrolling":
            with torch.no_grad():
                self.forward_iterations = 0
                no_grad_iterations = (
                    self.n_iterations if not self.training else self.n_iterations - 1
                )
                for i in range(no_grad_iterations):
                    self.forward_iterations = i + 1
                    new_h, self.reconstruction = self._nnmf_iteration(input)
                    self.convergence.append(F.mse_loss(new_h, self.h))
                    self.reconstruction_mse.append(
                        F.mse_loss(self.reconstruction, input)
                    )
                    self.h = new_h
                    if (
                        self.convergence_threshold > 0
                        and self.convergence[-1] < self.convergence_threshold
                    ):
                        break

            if self.training:
                for _ in range(self.unrolling_steps):
                    self.forward_iterations += 1
                    new_h, new_reconstruction = self._nnmf_iteration(input)
                    new_h = (
                        self.phantom_damping_factor * new_h
                        + (1 - self.phantom_damping_factor) * self.h
                    )
                    new_reconstruction = (
                        self.phantom_damping_factor * new_reconstruction
                        + (1 - self.phantom_damping_factor) * self.reconstruction
                    )
                    self.convergence.append(F.mse_loss(new_h, self.h))
                    self.reconstruction_mse.append(
                        F.mse_loss(self.reconstruction, input)
                    )
                    self.h = new_h
                    self.reconstruction = new_reconstruction

        elif self.backward_method == "david":
            self.h = self.david_backprop(
                input,
                self.h,
                self.n_iterations,
            )
            self.reconstruction = self._reconstruct(self.h, input=input)

        else:
            raise NotImplementedError(
                f"backward_method {self.backward_method} not implemented"
            )

        if self.return_reconstruction:
            return self.h, self.reconstruction
        else:
            return self.h

    @abstractmethod
    def david_backprop(self, input, h, n_iterations):
        raise NotImplementedError("David backprop not implemented for this layer")


class NNMFLayerDynamicWeight(NNMFLayer):
    def __init__(
        self,
        n_iterations,
        backward_method="all_grads",
        h_update_rate=1,
        w_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        convergence_threshold=0,
        normalize_input=False,
        normalize_input_dim=None,
        normalize_reconstruction=False,
        normalize_reconstruction_dim=None,
    ):
        NNMFLayer.__init__(
            self,
            n_iterations=n_iterations,
            backward_method=backward_method,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=return_reconstruction,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
        )
        self.w_update_rate = w_update_rate

    def _nnmf_iteration(self, input):
        new_h, new_reconstruction = super()._nnmf_iteration(input)
        self._update_weight(new_h, new_reconstruction, input)
        return new_h, new_reconstruction

    @abstractmethod
    def _update_weight(self, h, input):
        raise NotImplementedError


class NNMFDense(NNMFLayer):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="all_grads",
        convergence_threshold=0,
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        normalize_input=True,
        normalize_input_dim=-1,
        normalize_reconstruction=True,
        normalize_reconstruction_dim=-1,
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
            return_reconstruction,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.n_iterations = n_iterations

        self.weight = NonNegativeParameter(torch.rand(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=1)

    def _reset_h(self, x):
        h_shape = x.shape[:-1] + (self.out_features,)
        self.h = F.normalize(torch.ones(h_shape), p=1, dim=1).to(x.device)

    def _reconstruct(self, h, input=None, weight=None):
        if weight is None:
            weight = self.weight
        return F.linear(h, weight.t())

    def _forward(self, nnmf_update, weight=None):
        if weight is None:
            weight = self.weight
        return F.linear(nnmf_update, weight)

    def _process_h(self, h):
        h = self._secure_tensor(h)
        h = F.normalize(F.relu(h), p=1, dim=1)
        return h

    def jacobian(self, input, h):
        """
        Compute the jacobian of the forward pass with respect to h* (at the fixed point)

        returns I*(1-\sum_j (W_{ij}X_i/(\sum_k w_{kj}h_{bk}))) + h_{bi} * \sum_j ((w_{ij}X_{bj} w_{lj} )/(\sum_k w_{kj}h_{bk})²)
        """
        term1 = torch.diag_embed(1 - self._get_nnmf_update(input, h)[0])
        term2 = torch.einsum(
            "bi, ij, bj, lj -> bil",
            h,
            self.weight,
            input / self._reconstruct(h, input=input).pow(2),
            self.weight,
        )
        return term1 + term2

    def _check_forward(self, input):
        assert self.weight.sum(1, keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum(1)
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    @torch.no_grad()
    def normalize_weights(self):
        # weights may contain negative values after optimizer updates
        normalized_weight = F.normalize(self.weight.data, p=1, dim=-1)
        pos_weight = normalized_weight.clamp(min=SECURE_TENSOR_MIN)
        self.weight.data = F.normalize(pos_weight, p=1, dim=-1)

    def david_backprop(self, input, h, n_iterations):
        return FunctionalNNMFLinear.apply(input, self.weight, h, n_iterations, self._reconstruct, self._forward)


class NNMFDenseDynamicWeight(NNMFLayerDynamicWeight, NNMFDense):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
        backward_method="all_grads",
        convergence_threshold=0,
        h_update_rate=1,
        w_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
    ):
        NNMFDense.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            n_iterations=n_iterations,
            backward_method=backward_method,
            convergence_threshold=convergence_threshold,
            h_update_rate=h_update_rate,
            keep_h=keep_h,
            activate_secure_tensors=activate_secure_tensors,
            return_reconstruction=return_reconstruction,
        )
        self.w_update_rate = w_update_rate

    def _update_weight(self, h, reconstruction, input):
        nnmf_update = input / reconstruction
        new_weight = self.weight.data * F.linear(h.t(), nnmf_update.t())
        self.weight.data = (
            self.w_update_rate * new_weight
            + (1 - self.w_update_rate) * self.weight.data
        )
        self.normalize_weights()


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
        backward_method="all_grads",
        convergence_threshold=0,
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        normalize_input=True,
        normalize_input_dim=(1, 2, 3),
        normalize_reconstruction=True,
        normalize_reconstruction_dim=(1, 2, 3),
    ):
        super().__init__(
            n_iterations,
            backward_method,
            h_update_rate,
            keep_h,
            activate_secure_tensors,
            return_reconstruction,
            convergence_threshold=convergence_threshold,
            normalize_input=normalize_input,
            normalize_input_dim=normalize_input_dim,
            normalize_reconstruction=normalize_reconstruction,
            normalize_reconstruction_dim=normalize_reconstruction_dim,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.stride = _pair(stride)
        self.dilation = _pair(dilation)
        self.n_iterations = n_iterations
        self.normalize_channels = normalize_channels
        if self.dilation != (1, 1):
            raise NotImplementedError(
                "Dilation not implemented for NNMFConv2d, got dilation={self.dilation}"
            )

        self.weight = NonNegativeParameter(
            torch.rand(
                out_channels, in_channels, self.kernel_size[0], self.kernel_size[1]
            )
        )
        self.convolution_contribution_map = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.weight, a=0, b=1)
        self.weight.data = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))

    def normalize_weights(self):
        normalized_weight = F.normalize(self.weight.data, p=1, dim=(1, 2, 3))
        self.weight.data = F.normalize(
            normalized_weight.clamp(min=SECURE_TENSOR_MIN), p=1, dim=(1, 2, 3)
        )

    def _reconstruct(self, h, input=None, weight=None):
        if weight is None:
            weight = self.weight
        return F.conv_transpose2d(
            h,
            weight,
            padding=self.padding,
            stride=self.stride,
        )

    def _forward(self, nnmf_update, weight=None):
        if weight is None:
            weight = self.weight
        return F.conv2d(
            nnmf_update, weight, padding=self.padding, stride=self.stride
        )

    def _process_h(self, h):
        h = self._secure_tensor(h)
        if self.normalize_channels:
            h = F.normalize(F.relu(h), p=1, dim=1)
        return h

    def _reset_h(self, x):
        self.output_size = [
            (x.shape[-2] - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0]
            + 1,
            (x.shape[-1] - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1]
            + 1,
        ]
        reconstruct_size = [
            (self.output_size[0] - 1) * self.stride[0]
            - 2 * self.padding[0]
            + 1 * (self.kernel_size[0] - 1)
            + 1,
            (self.output_size[1] - 1) * self.stride[1]
            - 2 * self.padding[1]
            + 1 * (self.kernel_size[1] - 1)
            + 1,
        ]
        if reconstruct_size != list(x.shape[-2:]):
            raise ValueError(
                f"Reconstruction size {reconstruct_size} does not match input size {list(x.shape[-2:])}. Use ForwardNNMFConv2d instead"
            )
        self.h = torch.ones(x.shape[0], self.out_channels, *self.output_size).to(x.device)

    def _check_forward(self, input):
        assert self.weight.sum((1, 2, 3), keepdim=True).allclose(
            torch.ones_like(self.weight), atol=COMPARISSON_TOLERANCE
        ), self.weight.sum((1, 2, 3))
        assert (self.weight >= 0).all(), self.weight.min()
        assert (input >= 0).all(), input.min()

    def david_backprop(self, input, h, n_iterations):
        if self.convolution_contribution_map is None:
            # TODO: stride, padding, dilation
            self.convolution_contribution_map = torch.nn.functional.conv_transpose2d(
                torch.full(
                    (1, self.out_channels, *self.output_size),
                    1.0 / float(self.output_size[1]),
                    dtype=self.weight.dtype,
                    device=self.weight.device,
                ),
                torch.ones_like(self.weight),
                stride=1,
                padding=0,
                dilation=1,
            ) * (
                (input.shape[1] * input.shape[2] * input.shape[3])
                / (self.weight.shape[1] * self.weight.shape[2] * self.weight.shape[3])
            )
        return FunctionalNNMFConv2d.apply(input, self.weight, h, n_iterations, self._reconstruct, self._forward, self.convolution_contribution_map)


class ForwardNNMF(NNMFLayer):
    def _reconstruct(self, h, input, weight=None):
        return torch.autograd.functional.vjp(
            self._forward,
            input,
            self.h,
            create_graph=True,
        )[1]


class BackwardNNMF(NNMFLayer):
    def _forward(self, nnmf_update):
        return torch.autograd.functional.vjp(
            self._reconstruct,
            self.h,
            nnmf_update,
            create_graph=True,
        )[1]


class ForwardNNMFConv2d(ForwardNNMF, NNMFConv2d):
    """
    Forward NNMF for Conv2d layer
    """
