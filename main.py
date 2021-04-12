import torch
from torch import nn
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path


def get_data():
    data_path = Path("./data")
    fashion_path = data_path / "FashionMNIST"
    if not os.path.exists(fashion_path):
        os.makedirs(fashion_path)

    train_set = torchvision.datasets.FashionMNIST(
    root = data_path,
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))



if __name__ == "__main__":
    get_data()
