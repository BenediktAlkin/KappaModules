import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
# noinspection PyPackageRequirements
from tqdm import tqdm

from kappamodules.layers import DataNorm


def main():
    norm = DataNorm(dim=3)
    ds = CIFAR10(root=".", train=True, download=True, transform=ToTensor())
    dl = DataLoader(ds, batch_size=32)

    mean_deltas = []
    var_deltas = []
    for x, _ in tqdm(dl):
        pre_mean = norm.mean.clone()
        pre_var = norm.var.clone()
        norm(x)
        with torch.no_grad():
            mean_deltas.append((pre_mean - norm.mean).abs().mean())
            var_deltas.append((pre_var - norm.var).abs().mean())
    sd = norm.state_dict()
    print(f"mean: {sd['mean']}")
    print(f"var: {sd['var']}")
    print(f"std: {sd['var'].sqrt()}")

    plt.plot(range(len(mean_deltas)), mean_deltas, label="mean delta")
    plt.plot(range(len(var_deltas)), var_deltas, label="var delta")
    plt.legend()
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if os.name == "nt":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    main()
