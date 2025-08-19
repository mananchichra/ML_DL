import os
import sys
# Expand the '~' to the full path
mlp_path = os.path.expanduser('~/Downloads/SMAI_ASSIGNMENT/models/MLP')

# Insert the absolute path into sys.path
sys.path.insert(0, mlp_path)
from MLPReg import MLPRegressor



class AutoEncoder:
    def __init__(self, input_size, hidden_layers,output_size):
        self.encoder = MLPRegressor(input_size=input_size, hidden_layers=hidden_layers, output_size=output_size, learning_rate=0.001, activation='relu')
        self.decoder = MLPRegressor(input_size=output_size, hidden_layers=list(reversed(hidden_layers)), output_size=input_size, learning_rate=0.001, activation='relu')
        print(list(reversed(hidden_layers)))
    def fit(self, X, epochs=100, batch_size=None):
        # # Train the encoder
        # self.encoder.fit(X, X, X, X, epochs=epochs, batch_size=batch_size)

        # # Transform the data to the latent space
        # latent_representation = self.encoder.predict(X)

        # # Train the decoder
        # self.decoder.fit(latent_representation, X, latent_representation, X, epochs=epochs, batch_size=batch_size)
        # Step 1: Train the autoencoder (encoder + decoder) together
        for epoch in range(epochs):
            # Forward pass: Encoder compresses data, Decoder reconstructs data
            latent_representation = self.encoder.predict(X)
            reconstruction = self.decoder.predict(latent_representation)
            
            # Backward pass: Update both encoder and decoder
            self.decoder.backward(latent_representation, X)
            self.decoder.update_params()
            
            self.encoder.backward(X, latent_representation)
            self.encoder.update_params()
            
            if epoch % 10 == 0:
                reconstruction_loss = self.encoder.compute_loss(reconstruction, X)
                print(f"Epoch {epoch}, Reconstruction Loss: {reconstruction_loss}")

    def get_latent(self, X):
        return self.encoder.predict(X)

    def reconstruct(self, X):
        latent_representation = self.transform(X)
        return self.decoder.predict(latent_representation)