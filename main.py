# Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm


def FashionMNIST_dataset(train):
    data_path = Path("./data")
    fashion_path = data_path / "FashionMNIST"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(fashion_path):
        os.makedirs(fashion_path)
    
    if train==True:
        data_set = datasets.FashionMNIST(
            root = data_path,
            train = True,
            download = False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.286], [0.353]),
            ])
        )
    else:
        data_set = datasets.FashionMNIST(
            root = data_path,
            train = False,
            download = False,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.286], [0.353]),
            ])
        )

    data_loader = DataLoader(dataset=data_set, batch_size=64, shuffle=True)

    return data_set, data_loader


def get_mean_std(loader):
    """
    This function is borrowed from:
    https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_std_mean.py
    """

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    
    for data,_ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3]) # don't sum across channel dim
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5

    return mean, std
"""
_, train_loader = FashionMNIST_dataset(train=True)
mean, std = get_mean_std(train_loader) 
"""
#print(mean, std) 

class CNNetwork(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNNetwork, self).__init__()

        # in_challes = 1 due to greyscale

        # n_out = (n_in + 2*p - k)/s + 1
        # floor((28 + 2*1 - 3)/1) + 1 = 28

        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=8,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
        )

        self.maxpool2d = nn.MaxPool2d(
            kernel_size=(2,2),
            stride=(2,2),
            ceil_mode=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3,3),
            stride=(1,1),
            padding=(1,1),
        )

        # we use maxpool two times with padding with kernel 2 and stride 2, thus 
        # maxpool2d(28x28 pixel image) => 14x14 pixel image
        # maxpool2d(14x14 pixel image) => 7x7 pixel image, including 16 output channels

        self.hidden1 = nn.Linear(in_features=16*7*7, out_features=120)
        self.output = nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool2d(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2d(x)
        x = x.reshape(x.shape[0],-1) # alternatively x.reshape(-1,12*4*4)
        x = F.relu(self.hidden1(x))

        return self.output(x)


def train_model(model, dataloader, criterion, optimizer, epochs, device):

    for epoch in range(epochs):
        for batch_idx, (images, labels) in enumerate(dataloader):

            # Move to same device
            images = images.to(device=device)
            labels = labels.to(device=device)

            # Forward pass
            output = model(images)
            loss = criterion(output, labels)

            # Backprop
            optimizer.zero_grad()
            loss.backward()

            # Gradient descent step
            optimizer.step()

def eval_model(model, loader, device):      
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return (num_correct/num_samples)*100


if __name__ == "__main__":
    # Define configuration parameters
    config = dict()
    config["lr"] = 0.05
    config["batchsize_train"] = 64
    config["batchsize_test"] = 64
    config["epochs"] = 3
    config["use_cpu"] = torch.device("cpu")

    train_set, train_loader = FashionMNIST_dataset(train=True)
    test_set, test_loader = FashionMNIST_dataset(train=False)

    train_loader = DataLoader(dataset=train_set, batch_size = config["batchsize_train"], shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size = config["batchsize_test"], shuffle=True)

    model = CNNetwork().to(device=config["use_cpu"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, criterion, optimizer, config["epochs"], config["use_cpu"])
    train_eval = eval_model(model, train_loader, device=config["use_cpu"])
    test_eval = eval_model(model, test_loader, device=config["use_cpu"])
    print(f"Accuracy on training set: {train_eval:.2f}")
    print(f"Accuracy on test set: {test_eval:.2f}")

