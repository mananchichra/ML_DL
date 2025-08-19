import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, task='classification'):
        super(CNN, self).__init__()
        self.task = task

        # Define CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2_classification = nn.Linear(128, 10)  # For 10 classes
        self.fc2_regression = nn.Linear(128, 1)  # For regression

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))

        if self.task == 'classification':
            x = self.fc2_classification(x)
        elif self.task == 'regression':
            x = self.fc2_regression(x)

        return x
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

class CNN_tuning(nn.Module):
    def __init__(self, num_layers=2, dropout_rate=0.5, task='classification'):
        super().__init__()

        # Adjust number of convolutional layers
        self.convs = nn.ModuleList()
        in_channels = 1  # starting with grayscale input (assuming 1 channel for MNIST)
        for i in range(num_layers):
            self.convs.append(nn.Conv2d(in_channels, 32 * (i+1), kernel_size=3, padding=1))
            in_channels = 32 * (i+1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)

        # Calculate the size after conv layers by using a dummy input
        self._initialize_fc1_input_size()

        # Define fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 128)
        self.fc2 = nn.Linear(128, 10 if task == 'classification' else 1)

        self.task = task

    def _initialize_fc1_input_size(self):
        # Create a dummy input with the same dimensions as your input data
        dummy_input = torch.zeros(1, 1, 28, 28)  # MNIST images are 28x28
        x = dummy_input
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))

        # Flatten x to find the size for fc1
        self.fc1_input_size = x.numel()  # This gives total number of elements

    def forward(self, x):
        # Forward through convolutional layers
        for conv in self.convs:
            x = self.pool(F.relu(conv(x)))

        # Flatten the output
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        if self.task == 'classification':
            return F.log_softmax(x, dim=1)
        else:
            return x  # For regression, no activation applied
