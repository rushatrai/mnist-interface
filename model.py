import os
import time

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim

from tqdm import tqdm


def get_mean_std(loader):
  '''
  Calculates mean and std of input images.

  Args:
    loader (torch.DataLoader): Loader with images

  Returns:
    mean (torch.Tensor): Mean of images in loader
    std (torch.Tensor): Standard deviation of images in loader
  '''
  channels_sum, channels_squared_sum, num_batches = 0, 0, 0

  for data, _ in loader:
    channels_sum += torch.mean(data, dim=[0,2,3])  # mean across [no. of examples, height, width]
    channels_squared_sum += torch.mean(data**2, dim=[0,2,3])  # squared mean across [no. of examples, height, width]
    num_batches += 1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/(num_batches-mean**2))**0.5

    return mean, std


class Net(nn.Module):
    '''
    model definition
    '''

    def __init__(self):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.25),
            nn.Flatten(),
        )
        self.layer5 = nn.Sequential(
            nn.Linear(576, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.Linear(128, 84, bias=False),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.layer8 = nn.Sequential(
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = transforms.Normalize(mean, std)(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        return x


# downloads and loads MNIST train set
transform = transforms.Compose([transforms.ToTensor(), transforms.RandomAffine(degrees=10, translate=(0.1,0.1))])
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, pin_memory=True)

# downloads and loads MNIST test set
val_data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
val_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False, pin_memory=True)

# uses GPU if available
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"

device = torch.device(dev)

# gets mean and std of dataset
mean, std = get_mean_std(train_loader)


def run_model():
    # defines parameters
    model = Net().to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=2)
    criterion = nn.NLLLoss()

    # iterates through epochs
    for epoch in range(30):
        print(f"\nEpoch {epoch+1}/{30}.")

        # train loop
        model.train()

        total_train_loss = 0
        total_correct = 0

        for i, (images, labels) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Calculates train accuracy
            outputs_probs = nn.functional.softmax(
                outputs, dim=1)  # gets probabilities
            for idx, preds in enumerate(outputs_probs):
                # if label with max probability matches true label
                if labels[idx] == torch.argmax(preds.data):
                    total_correct += 1

        train_loss = total_train_loss/(i+1)
        train_accuracy = total_correct/len(train_data)

        print(f"Train set:- Loss: {train_loss}, Accuracy: {train_accuracy}.")

        # saves model state
        if not os.path.exists("./saved_models"):
            os.mkdir("./saved_models")
        torch.save(model.state_dict(), f"./saved_models/mnist-cnn-{time.time()}.pt")

        # val loop
        model.eval()

        total_val_loss = 0
        total_correct = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(tqdm(val_loader)):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                outputs_probs = nn.functional.softmax(outputs, dim=1)
                for idx, preds in enumerate(outputs_probs):
                    if labels[idx] == torch.argmax(preds.data):
                        total_correct += 1

        val_loss = total_val_loss/(i+1)
        val_accuracy = total_correct/len(val_data)

        print(f"Val set:- Loss: {val_loss}, Accuracy: {val_accuracy}.")

        # adjusts lr
        scheduler.step(val_loss)
