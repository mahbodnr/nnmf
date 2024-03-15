# %%
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from nnmf.modules import NNMFDense
# %%
# load CIFAR-10 dataset
batch_size = 100
data_path = "~/data"
cifar10 = datasets.CIFAR10(
    data_path,
    train=True,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: F.normalize(x, dim=None)),
        ]
    ),
)
dataloader = DataLoader(cifar10, batch_size=100, shuffle=True)
input, labels = next(iter(dataloader))
# input = input.requires_grad_(True)
# %%
def get_gradients(backward_method, iterations, input):
    torch.manual_seed(9248) # 92:48
    nnmf = NNMFDense(
        in_features=32*32*3,
        out_features=10,
        n_iterations=iterations,
        backward_method=backward_method,
        activate_secure_tensors=False,
        normalize_reconstruction=False,
    )
    nnmf.weight.data = torch.rand(10, 32*32*3)
    output = nnmf(input.view(batch_size, -1))
    loss = F.cross_entropy(output, labels)
    loss.backward()
    return nnmf.weight.grad.clone(), input.grad.clone()
# %% 
iterations = 100
# All gradients
wg_all_grads, ig_all_grads = get_gradients("all_grads", iterations, input.clone().requires_grad_(True))
print(
    f"""All gradients:
    Weight gradients: mean={wg_all_grads.mean()}, std={wg_all_grads.std()}
    Input gradients: mean={ig_all_grads.mean()}, std={ig_all_grads.std()}
"""
)
# Only the last iteration
wg_last_iter, ig_last_iter = get_gradients("last_iter", iterations, input.clone().requires_grad_(True))
print(
    f"""Only the last iteration:
    Weight gradients: mean={wg_last_iter.mean()}, std={wg_last_iter.std()}
    Input gradients: mean={ig_last_iter.mean()}, std={ig_last_iter.std()}
"""
)
#  Implicit differentiation
wg_implicit, ig_implicit = get_gradients("implicit", iterations, input.clone().requires_grad_(True))
print(
    f"""Implicit differentiation:
    Weight gradients: mean={wg_implicit.mean()}, std={wg_implicit.std()}
    Input gradients: mean={ig_implicit.mean()}, std={ig_implicit.std()}
"""
)
# %% 
wg_error = {
    "implicit": [],
    "last_iter": [],
    "last_iter_scaled": [],
}

iteration_range = torch.arange(5, 100, 10)
for iterations in iteration_range:
    iterations = int(iterations)
    wg_all_grads, ig_all_grads = get_gradients("all_grads", iterations, input.clone().requires_grad_(True))
    wg_last_iter, ig_last_iter = get_gradients("last_iter", iterations, input.clone().requires_grad_(True))
    wg_implicit, ig_implicit = get_gradients("implicit", iterations, input.clone().requires_grad_(True))

    wg_error["implicit"].append((wg_implicit - wg_all_grads).abs().mean())
    wg_error["last_iter"].append((wg_last_iter - wg_all_grads).abs().mean())
    wg_error["last_iter_scaled"].append((wg_last_iter * iterations**0.5 - wg_all_grads).abs().mean())
    # all vs last iter
    plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f"All gradients vs only the last iteration. Iterations: {iterations}", fontsize=16)
    plt.subplot(2, 1, 1)
    plt.title("Weight gradients")
    alpha = 0.8
    plt.hist(wg_all_grads.flatten().detach().numpy(), bins=100, alpha=alpha, label="All gradients", color="green")
    plt.hist(wg_last_iter.flatten().detach().numpy() * iterations**0.5, bins=100, alpha=alpha, label="Last iteration scaled", color="orange")
    plt.hist(wg_last_iter.flatten().detach().numpy(), bins=100, alpha=alpha, label="Last iteration", color="red")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Input gradients")
    plt.hist(ig_all_grads.flatten().detach().numpy(), bins=100, alpha=alpha, label="All gradients", color="green")
    plt.hist(ig_last_iter.flatten().detach().numpy() * iterations**0.5, bins=100, alpha=alpha, label="Last iteration scaled", color="orange")
    plt.hist(ig_last_iter.flatten().detach().numpy(), bins=100, alpha=alpha, label="Last iteration", color="red")

    plt.tight_layout()
    plt.savefig(f"media/all_vs_last_iter_iter_{iterations}.png")
    
    # all vs implicit
    plt.subplots(2, 1, figsize=(10, 10))
    plt.suptitle(f"Iterations: {iterations}", fontsize=16)
    plt.subplot(2, 1, 1)
    plt.title("Weight gradients")
    alpha = 0.5
    plt.hist(wg_implicit.flatten().detach().numpy(), bins=100, alpha=alpha, label="Implicit differentiation", color="skyblue")
    plt.hist(wg_all_grads.flatten().detach().numpy(), bins=100, alpha=alpha, label="All gradients", color="green")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Input gradients")
    plt.hist(ig_implicit.flatten().detach().numpy(), bins=100, alpha=alpha, label="Implicit differentiation", color="skyblue")
    plt.hist(ig_all_grads.flatten().detach().numpy(), bins=100, alpha=alpha, label="All gradients", color="green")

    plt.tight_layout()
    plt.savefig(f"media/all_vs_implicit_iter_{iterations}.png")

    plt.close("all")

# plot error
plt.plot(iteration_range, wg_error["implicit"], label="Implicit differentiation")
plt.plot(iteration_range, wg_error["last_iter"], label="Last iteration")
plt.plot(iteration_range, wg_error["last_iter_scaled"], label="Last iteration scaled")
plt.xlabel("Iterations")
plt.ylabel("Mean absolute error")
plt.title("Weight gradients error")
plt.legend()

plt.savefig("media/weight_gradients_error.png")
# %%
