from implicit_backprop.modules import NNMFDense as NNMFDenseImplicit
from nnmf.modules import NNMFDense as NNMFDense

import torch

input = torch.rand(512, 128)
input = input.cuda()

nnmf = NNMFDense(
    in_features=128,
    out_features=10,
    n_iterations=100,
).cuda()

nnmf_implicit = NNMFDenseImplicit(
    in_features=128,
    out_features=10,
    n_iterations=100,
).cuda()

# nnmf_implicit.weight.data = nnmf.weight.data

output = nnmf(input)
output_implicit = nnmf_implicit(input)

# print("NNMF output: ", output)
# print("NNMF implicit output: ", output_implicit)

import matplotlib.pyplot as plt

plt.hist(output.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF")
plt.hist(output_implicit.cpu().detach().numpy().flatten(), bins=100, alpha=0.5, label="NNMF implicit")
plt.legend()
plt.show()
