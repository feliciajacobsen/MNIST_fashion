# Imports
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path


def FashionMNIST_dataset(train):
    data_path = Path("./data")
    fashion_path = data_path / "FashionMNIST"
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    if not os.path.exists(fashion_path):
        os.makedirs(fashion_path)
    train_set = torchvision.datasets.FashionMNIST(
        root = data_path,
        train = train,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ])
    )
    return train_set


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, # greyscale image
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

        # flatten tensor so it can be passed to dense layer
        x = F.relu(self.hidden1(x.reshape(-1,12*4*4)))

        x = F.relu(self.hidden2(x))

        # dont add softmax of output since we use CrossEntropy later
        return self.output(x)

def train(model, dataloader, criterion, optimizer, epochs, device):
    #model.to(device)

    dataiter = iter(dataloader)
    images, labels = dataiter.next()

    losses = []
    for epoch in range(epochs):
        for images, labels in dataloader:
            # Flatten image
            images = images.view(images.shape[0],-1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
    return np.mean(losses)

def train_eval(model, dataloader_train, dataloader_test, criterion, optimizer, epochs, device):
    dataiter = iter(dataloader)
    images, labels = dataiter.next()
    train_losses, test_losses = [], []

    for epoch in range(epochs):
        model.train()
        losses = []
        for images, labels in trainloader:
            # Flatten Fashion-MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)

            # Training pass
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_losses.append(loss.item())
        else:
            test_losses = []
            accuracy = []

            # Turn off gradients for validation, saves memory and computation
            with torch.no_grad():
              # Set the model to evaluation mode
              model.eval()

              # Validation pass
              for images, labels in dataloader_test:
                  images = images.view(images.shape[0], -1)
                  log_ps = model(images)
                  test_losses.append(criterion(log_ps, labels))

                  ps = torch.exp(log_ps)
                  top_p, top_class = ps.topk(1, dim = 1)
                  equals = top_class == labels.view(*top_class.shape)
                  accuracy.append(torch.mean(equals.type(torch.FloatTensor)))

            model.train()
            train_losses.append(losses/len(dataloader_train))
            test_losses.append(test_losses/len(dataloader_test))

            print("Epoch: {}/{}..".format(epoch+1, epochs),
                  "Training loss: {:.3f}..".format(running_losses/len(dataloader_train)),
                  "Test loss: {:.3f}..".format(test_losses/len(dataloader_test)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(dataloader_test)))






if __name__ == "__main__":
    # Define configuration parameters
    config = dict()
    config["lr"] = 0.05
    config["momentum"] = 0.9
    config["num_classes"] = 10
    config["batchsize_train"] = 64
    config["batchsize_test"] = 64
    config["epochs"] = 3
    config["use_cpu"] = torch.device("cpu")

    train_dataset = FashionMNIST_dataset(train=True)
    test_dataset = FashionMNIST_dataset(train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size = config["batchsize_train"], shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size = config["batchsize_test"], shuffle=True)

    model = Network()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, criterion, optimizer, config["epochs"], config["use_cpu"])
    train_eval(model, train_loader, test_loader, criterion, optimizer, config["epochs"], config["use_cpu"])
