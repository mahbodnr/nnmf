from abc import abstractmethod
from typing import Union, List
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

from implicit_backprop.autograd import ImplicitGradient

COMPARISSON_TOLERANCE = 1e-5
SECURE_TENSOR_MIN = 1e-5


class NNMFLayer(nn.Module):
    def __init__(
        self,
        n_iterations,
        h_update_rate=1,
        keep_h=False,
        activate_secure_tensors=True,
        return_reconstruction=False,
        convergence_threshold=0,
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

    def _secure_tensor(self, t):
        return t.clamp_min(SECURE_TENSOR_MIN) if self.activate_secure_tensors else t

    @abstractmethod
    def normalize_weights(self):
        raise NotImplementedError

    @abstractmethod
    def _reset_h(self, x):
        raise NotImplementedError

    @abstractmethod
    def _reconstruct(self, h, input= None, weight=None):
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

    def _get_nnmf_update(self, input, h, weight=None):
        reconstruction = self._reconstruct(h, input=input, weight=weight)
        reconstruction = self._secure_tensor(reconstruction)
        if self.normalize_reconstruction:
            reconstruction = F.normalize(
                reconstruction, p=1, dim=self.normalize_reconstruction_dim, eps=1e-20
            )
        return self._forward(input / reconstruction, weight=weight), reconstruction

    def _nnmf_iteration(self, input, weight=None):
        nnmf_update, reconstruction = self._get_nnmf_update(input, self.h, weight=weight)
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

        self.h = ImplicitGradient.apply(input, self.weight, self.h, self.n_iterations, self._nnmf_iteration)

        return self.h



class NNMFDense(NNMFLayer):
    def __init__(
        self,
        in_features,
        out_features,
        n_iterations,
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

        self.weight = torch.nn.Parameter(torch.rand(out_features, in_features))
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


if __name__ == "__main__":
    layer = NNMFDense(in_features= 5, out_features= 4, n_iterations= 2).cuda()
    input = torch.rand(10, 5).requires_grad_().cuda()
    output = layer(input)
    print(output)
    l = output.sum()
    l.backward()

    print(layer.weight.grad)
    print(input.grad)

    # gradcheck
    from torch.autograd import gradcheck

    input = torch.rand(10, 5).requires_grad_().cuda()
    layer = NNMFDense(in_features= 5, out_features= 4, n_iterations= 100).cuda()
    test = gradcheck(layer, (input,), eps=1e-6, atol=1e-4)
    print(test)
