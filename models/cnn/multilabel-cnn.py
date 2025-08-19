import torch
import torch.nn as nn
class MultiLabelCNN(nn.Module):
    def __init__(self, num_conv_layers=3, dropout_rate=0.5, max_digits=5, num_classes=10):
        super(MultiLabelCNN, self).__init__()

        self.num_conv_layers = num_conv_layers
        self.dropout_rate = dropout_rate
        self.max_digits = max_digits
        self.num_classes = num_classes

        # convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        # Fully Connected Layer
        if self.num_conv_layers == 2:
            self.fc = nn.Linear(64 * 7 * 7, num_classes * max_digits)
        else:
            self.fc = nn.Linear(128 * 3 * 3, num_classes * max_digits)

        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        if self.num_conv_layers > 1:
            x = F.relu(self.conv2(x))
            x = self.pool(x)
        if self.num_conv_layers > 2:
            x = F.relu(self.conv3(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc(x)))
        return torch.sigmoid(x).view(-1, self.num_classes, self.max_digits)

