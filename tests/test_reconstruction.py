# %%
import unittest

import torch
from nnmf.modules import NNMFDense, NNMFConv2d
from functools import partial
# %%
def check_forward(layer, input):
    forw = layer._forward(input)
    rec_jac = torch.autograd.functional.vjp(
        partial(layer._reconstruct, input=input),
        layer.h,
        input,
    )[1]
    return torch.allclose(forw, rec_jac)


def check_backward(layer, input):
    forw_jac = torch.autograd.functional.vjp(
        layer._forward,
        input,
        layer.h,
    )[1]
    rec = layer._reconstruct(layer.h, input=input)
    return torch.allclose(forw_jac, rec)


class TestReconstructCheck(unittest.TestCase):
    def test_nnmf_dense(self):
        nnmf = NNMFDense(
            in_features=128,
            out_features=100,
            n_iterations=10,
        )
        input = torch.rand(64, 128)
        input = nnmf._prepare_input(input)
        nnmf._reset_h(input)
        self.assertTrue(check_forward(nnmf, input))
        self.assertTrue(check_backward(nnmf, input))

    def test_nnmf_conv_1(self):
        nnmf = NNMFConv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=6,
            stride=2,
            padding=1,
            n_iterations=10,
        )
        input = torch.rand(64, 3, 32, 32)
        input = nnmf._prepare_input(input)
        nnmf._reset_h(input)
        self.assertTrue(check_forward(nnmf, input))
        self.assertTrue(check_backward(nnmf, input))

    def test_nnmf_conv_2(self):
        """
        NNMFConv2d will fail this test.
        """
        nnmf = NNMFConv2d(
            in_channels=3,
            out_channels=10,
            kernel_size=3,
            stride=4,
            padding=0,
            n_iterations=10,
        )
        input = torch.rand(64, 3, 32, 32)
        nnmf._reset_h(input)
        self.assertTrue(check_forward(nnmf, input))
        self.assertTrue(check_backward(nnmf, input))

if __name__ == "__main__":
    unittest.main()

# %%
