class MLP_Autoencoder(nn.Module):
    def __init__(self, input_size=784, hidden_dims=[256, 128], latent_dim=32):
        super(MLP_Autoencoder, self).__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_size
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
        encoder_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        encoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_size))
        decoder_layers.append(nn.Sigmoid())  # to match the pixel range of (0, 1)

        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the images
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

class MLP_Classifier(nn.Module):
    def __init__(self, input_dim=32, hidden_dims=[128, 64], output_dim=10):
        super(MLP_Classifier, self).__init__()

        classifier_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            classifier_layers.append(nn.Linear(in_dim, h_dim))
            classifier_layers.append(nn.ReLU())
            in_dim = h_dim
        classifier_layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        return self.classifier(x)
