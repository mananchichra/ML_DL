## KNN
    def accuracy(self, y_true):
        correct = np.sum(y_true == self.y_pred)
        return correct / len(y_true)

    def precision(self, y_true):
        precision_scores = []
        for cls in np.unique(y_true):
            tp = np.sum((y_true == cls) & (self.y_pred == cls))
            fp = np.sum((y_true != cls) & (self.y_pred == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_scores.append(precision)
        macro_precision = np.mean(precision_scores)
        return macro_precision

    def recall(self, y_true):
        recall_scores = []
        for cls in np.unique(y_true):
            tp = np.sum((y_true == cls) & (self.y_pred == cls))
            fn = np.sum((y_true == cls) & (self.y_pred != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_scores.append(recall)
        macro_recall = np.mean(recall_scores)
        return macro_recall

    def f1_score(self, y_true, average='macro'):
        precision = self.precision(y_true)
        recall = self.recall(y_true)
        if average == 'macro':
            macro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            return macro_f1
        elif average == 'micro':
            tp = np.sum((y_true == self.y_pred))
            fp = np.sum((y_true != self.y_pred))
            micro_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            micro_recall = tp / (tp + fp) if (tp + fp) > 0 else 0
            micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            return micro_f1
        else:
            raise ValueError("average must be 'macro' or 'micro'")


## GMM

    def get_likelihood(self, X):
        log_likelihoods = np.array([
            np.log(self.weights_[j]) + multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covariances_[j])
            for j in range(self.n_components)
        ]).T
        
        # Use log-sum-exp trick for numerical stability
        max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
        log_likelihoods -= max_log_likelihood
        
        return np.sum(np.log(np.sum(np.exp(log_likelihoods), axis=1)) + max_log_likelihood.flatten())

    def aic(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.get_likelihood(X)
        # Number of parameters: means + covariances + weights
        n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
        aic = 2 * n_params - 2 * log_likelihood
        return aic

    def bic(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.get_likelihood(X)
        # Number of parameters: means + covariances + weights
        n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        return bic

## PCA
    def checkPCA(self, X):
        mse_threshold = False
        X_transformed = self.transform(X)
        X_recon = self.inverse_transform(X_transformed)

        mse = np.mean((X - X_recon) ** 2)
        print(f"Mean Square Error: {mse:.4f}")
        
        # Ensure that cumulative explained variance is non-decreasing
        if not np.all(np.diff(self.cum_explained_variance) >= 0):
            return False
        
        # Check if MSE is below a given threshold
        threshold = 0.2
        if mse < threshold:
            mse_threshold = True
        
        # Return True if both MSE is below the threshold and number of components match
        return mse_threshold and X_transformed.shape[1] == self.n_components



## MLPReg

    def compute_loss(self, predictions, y):
    return np.mean((predictions - y) ** 2)  # Mean Squared Error (MSE)

    def compute_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_rmse(self, y_true, y_pred):
        return np.sqrt(self.compute_mse(y_true, y_pred))

    def compute_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    
    def predict(self, X):
        return self.forward(X)

## MLP-Binary Classification

        val_loss = self._compute_loss(activations_val, y_val)
        val_predictions = (activations_val[-1] > 0.5).astype(int)  
        val_accuracy = np.mean(val_predictions.T == y_val)