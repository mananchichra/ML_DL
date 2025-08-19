class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        # Encoder (CNN)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),  # (32, 32, 128)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)  # (64, 16, 64)
        )
        self.fc = nn.Linear(64 * 16 * 64, 128)

        # Decoder (RNN)
        self.rnn = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(256, num_classes)  # 128*2 (bidirectional)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1)  # Flatten
        x = self.fc(x)
        x = x.unsqueeze(1).repeat(1, MAX_WORD_LEN, 1)  # Repeat for sequence length

        x, _ = self.rnn(x)
        x = self.output_layer(x)  # (batch_size, MAX_WORD_LEN, num_classes)