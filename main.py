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

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=6,
            kernel_size=5
        )

        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=12,
            kernel_size=5
        )

        self.hidden1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.hidden2 = nn.Linear(in_features=120, out_features=60)
        self.output = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)

        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=2, stride=2)

        # flatten tensor so it can be passed to dense hidden layer
        x = F.relu(self.hidden1(x.reshape(-1,12*4*4)))

        x = F.relu(self.hidden2(x))

        # dont add softmax of output since we use CrossEntropy later
        return self.output(x)


if __name__ == "__main__":
    get_data()

    config = dict()
    config["lr"] = 0.05
    config["batchsize_train"] = 64
    config["epochs"] = 3
