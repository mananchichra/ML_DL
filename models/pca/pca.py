
# import numpy as np
# class PCA:
    
#     def __init__(self, n_components):
#         print("initial")
#         self.n_components = n_components   
        
#     def fit(self, X):
#         # Standardize data 
#         X = X.copy()
#         self.mean = np.mean(X, axis = 0)
#         self.scale = np.std(X, axis = 0)
#         X_std = (X - self.mean) / self.scale
        
#         # Eigendecomposition of covariance matrix       
#         cov_mat = np.cov(X_std.T)
#         eig_vals, eig_vecs = np.linalg.eigh(cov_mat) 
    
#         # Adjusting the eigenvectors that are largest in absolute value to be positive    
#         max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
#         signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
#         eig_vecs = eig_vecs*signs[np.newaxis,:]
#         eig_vecs = eig_vecs.T
       
#         eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i,:]) for i in range(len(eig_vals))]
#         eig_pairs.sort(key=lambda x: x[0], reverse=True)
#         eig_vals_sorted = np.array([x[0] for x in eig_pairs])
#         eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        
#         self.components = eig_vecs_sorted[:self.n_components,:]
        
#         # Explained variance ratio
#         self.explained_variance_ratio = [i/np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
#         self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

#         return self

#     def transform(self, X):
#         X = X.copy()
#         X_mean = np.mean(X, axis = 0)
#         X_std = (X - X_mean)
#         X_proj = X_std.dot(self.components.T)
        
#         return X_proj
    
#     # def checkPCA(self, X):
#     #     X_transformed = self.transform(X)
#     #     return X_transformed.shape[1] == self.n_components
    
#     def checkPCA(self, X):
        
#         mse_threshold = False
#         X_transformed = self.transform(X)
#         X_recon = self.inverse_transform(X_transformed)

#         mse = np.mean((X - X_recon) ** 2)
#         print(f"Mean Square Error: {mse:.4f}")
        
#         if not np.all(np.diff(self.cum_explained_variance) >= 0):
#             return False
        
#         threshold = 0.2
        
#         if mse < threshold:
#             mse_threshold = True
#         return mse_threshold and X_transformed.shape[1] == self.n_components
import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components   
        
    def fit(self, X):
        # Standardize data 
        X = X.copy()

        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        X_std = (X - self.mean) / self.scale
        
        # Eigendecomposition of covariance matrix       
        cov_mat = np.cov(X_std.T)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
    
        # Adjusting the eigenvectors that are largest in absolute value to be positive    
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        signs = np.sign(eig_vecs[max_abs_idx, range(eig_vecs.shape[0])])
        eig_vecs = eig_vecs * signs[np.newaxis, :]
        eig_vecs = eig_vecs.T
       
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[i, :]) for i in range(len(eig_vals))]
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        eig_vals_sorted = np.array([x[0] for x in eig_pairs])
        eig_vecs_sorted = np.array([x[1] for x in eig_pairs])
        
        self.components = eig_vecs_sorted[:self.n_components, :]
        
        # Explained variance ratio
        self.explained_variance_ratio = [i / np.sum(eig_vals) for i in eig_vals_sorted[:self.n_components]]
        
        self.cum_explained_variance = np.cumsum(self.explained_variance_ratio)

        return self

    def transform(self, X):
        X = X.copy()
        X_std = (X - self.mean) / self.scale
        X_proj = X_std.dot(self.components.T)
        return X_proj

    def inverse_transform(self, X_transformed):
        """Reconstruct the original data from the reduced dimensions"""
        X_recon = X_transformed.dot(self.components) * self.scale + self.mean
        return X_recon
    
    def checkPCA(self, X):
        """Check if PCA transformation is valid by comparing reconstruction error and cumulative explained variance"""
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
