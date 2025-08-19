import torch
import torch.nn as nn
import torch.optim as optim

class CnnAutoencoder(nn.Module):
    def __init__(self, input_channels=1, num_filters=(32, 64, 128), kernel_size=3, latent_dim=128,
                 learning_rate=0.001, optimizer_choice='adam'):
        super(CnnAutoencoder, self).__init__()

        self.learning_rate = learning_rate
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.optimizer_choice = optimizer_choice
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for out_channels in num_filters:
            encoder_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2)
            )
            encoder_layers.append(nn.ReLU())
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        # Decoder
        decoder_layers = []
        for out_channels in reversed(num_filters):
            decoder_layers.append(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, output_padding=1)
            )
            decoder_layers.append(nn.ReLU())
            in_channels = out_channels

        # Update the last layer to use Tanh instead of Sigmoid
        decoder_layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        decoder_layers.append(nn.Tanh())  # Changed from Sigmoid to Tanh
        self.decoder = nn.Sequential(*decoder_layers)

        # decoder_layers = []
        # for out_channels in reversed(num_filters):
        #     decoder_layers.append(
        #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=kernel_size // 2, output_padding=1)
        #     )
        #     decoder_layers.append(nn.ReLU())
        #     in_channels = out_channels
        # decoder_layers.append(nn.ConvTranspose2d(in_channels, input_channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2))
        # decoder_layers.append(nn.Sigmoid())
        # self.decoder = nn.Sequential(*decoder_layers)

        # Initialize optimizer based on optimizer_choice
        self.optimizer = self._initialize_optimizer()

    def _initialize_optimizer(self):
        if self.optimizer_choice.lower() == 'adam':
            return optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_choice.lower() == 'sgd':
            return optim.SGD(self.parameters(), lr=self.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer choice: {self.optimizer_choice}")

    def encode(self, x):
        x = self.encoder(x)
        self.encoded_shape = x.shape  # Store shape for decoding
        flattened_size = x.size(1) * x.size(2) * x.size(3)

        # Dynamically create fc_upsample with the correct size
        self.fc_upsample = nn.Linear(flattened_size, self.latent_dim).to(x.device)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layer
        x = self.fc_upsample(x)     # Compress to latent_dim
        return x

    def decode(self, x):
        # Define fc_downsample to expand from latent_dim back to flattened size
        if not hasattr(self, 'fc_downsample'):
            flattened_size = self.encoded_shape[1] * self.encoded_shape[2] * self.encoded_shape[3]
            self.fc_downsample = nn.Linear(self.latent_dim, flattened_size).to(x.device)

        x = self.fc_downsample(x)
        x = x.view(x.size(0), *self.encoded_shape[1:])  # Reshape to encoded shape
        x = self.decoder(x)
        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed
