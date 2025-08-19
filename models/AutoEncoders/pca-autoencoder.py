class PcaAutoencoder:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.pca = None

    def fit(self, X):
        """
        Fit the PCA model to the training data.
        """
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)

    def encode(self, X):
        """
        Encode data using the PCA model.
        """
        if self.pca is None:
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        return self.pca.transform(X)

    def forward(self, X):
        """
        Reconstruct data from the encoded representation.
        """
        if self.pca is None:
            raise ValueError("The model has not been fitted yet. Call fit() first.")
        encoded = self.encode(X)
        return self.pca.inverse_transform(encoded)
