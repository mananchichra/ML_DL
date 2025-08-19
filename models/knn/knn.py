from math import sqrt
import numpy as np
from collections import Counter


import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from tqdm import tqdm



class Metrics:
    @staticmethod
    def accuracy(y_true, y_pred):
        """
        Calculates the accuracy score.
        
        Parameters:
        y_true (numpy.ndarray): True labels.
        y_pred (numpy.ndarray): Predicted labels.
        
        Returns:
        float: Accuracy score.
        """
        return np.sum(y_true == y_pred) / len(y_true)

    @staticmethod
    def precision(y_true, y_pred, average='macro'):
        classes = np.unique(y_true)
        precision_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            precision_scores.append(precision)
        
        if average == 'macro':
            return np.mean(precision_scores)
        elif average == 'micro':
            tp_sum = np.sum([(y_true == cls) & (y_pred == cls) for cls in classes])
            fp_sum = np.sum([(y_true != cls) & (y_pred == cls) for cls in classes])
            return tp_sum / (tp_sum + fp_sum)
        else:
            raise ValueError("Invalid average method")

    @staticmethod
    def recall(y_true, y_pred, average='macro'):
        classes = np.unique(y_true)
        recall_scores = []
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            recall_scores.append(recall)
        
        if average == 'macro':
            return np.mean(recall_scores)
        elif average == 'micro':
            tp_sum = np.sum([(y_true == cls) & (y_pred == cls) for cls in classes])
            fn_sum = np.sum([(y_true == cls) & (y_pred != cls) for cls in classes])
            return tp_sum / (tp_sum + fn_sum)
        else:
            raise ValueError("Invalid average method")

    @staticmethod
    def f1_score(y_true, y_pred, average='macro'):
        precision = Metrics.precision(y_true, y_pred, average)
        recall = Metrics.recall(y_true, y_pred, average)
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        else:
            return 0


class KNN:
    def _init_(self, k, metric='euclidean'):
        self.k = k
        self.metric = metric
    def set_params(self, k=None, distance_metric=None):
        if k is not None:
            self.k = k
        if distance_metric is not None:
            self.metric = distance_metric
        
    def save_model(self, file_name):
        np.save(file_name, self.coefficients)
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = []
        for dpoint in X_test:
            distances = self._compute_distances(dpoint)
            sorted_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = self.y_train[sorted_indices]
            most_common = pd.Series(k_nearest_labels).mode().values[0]
            predictions.append(most_common)
        return np.array(predictions)
    
    def _compute_distances(self, x):
        distances = []
        for x_train in self.X_train:
            if self.metric == 'manhattan':
                dist = np.sum(np.abs(x_train - x))
            elif self.metric == 'euclidean':
                dist = np.sqrt(np.sum((x_train - x) ** 2))
            else:
                raise ValueError(f"Unknown distance metric: {self.metric}")
            distances.append(dist)
        return distances
    
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

class KNNVectorized:
    def _init_(self, k, metric='euclidean', batch_size=None):
        self.k = k
        self.metric = metric
        self.batch_size = batch_size
    def set_params(self, k=None, distance_metric=None):
        if k is not None:
            self.k = k
        if distance_metric is not None:
            self.metric = distance_metric
    
    def save_model(self, file_name):
        np.save(file_name, self.coefficients)
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size  
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        num_samples = X_test.shape[0]
        predictions = np.zeros(num_samples, dtype=int)
        
        if self.batch_size is None:
            self.batch_size = num_samples

        
        for start in tqdm(range(0, num_samples, self.batch_size), desc="Processing batches", unit="batch"):
            end = min(start + self.batch_size, num_samples)
            X_test_batch = X_test[start:end]
            predictions[start:end] = self._predict_batch(X_test_batch)
        
        self.y_pred = predictions
        return self.y_pred

 
    def _predict_batch(self, X_test_batch):
    # Vectorized distance calculation
        if self.metric == 'euclidean':
            dists = np.linalg.norm(X_test_batch[:, np.newaxis] - self.X_train, axis=2)
        elif self.metric == 'manhattan':
            dists = np.sum(np.abs(X_test_batch[:, np.newaxis] - self.X_train), axis=2)
        else:
            raise ValueError(f"Unknown distance metric: {self.metric}")
    
    # Find the k nearest neighbors
        nearest_indices = np.argpartition(dists, self.k, axis=1)[:, :self.k]
        nearest_labels = self.y_train[nearest_indices]
    
    # Use np.apply_along_axis to apply bincount in a vectorized manner
        def most_common_label(labels):
            return np.bincount(labels).argmax()
    
    # Apply most_common_label across the rows
        predictions = np.apply_along_axis(most_common_label, 1, nearest_labels)

        return predictions


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
        
        
class KNNPerformanceEvaluator(KNNVectorized,KNN):
    def __init__(self, X_train, y_train, X_test, y_test):
        # Ensure all inputs are numpy arrays
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.models = {}

    def add_model(self, model_name, model_instance):
        self.models[model_name] = model_instance

    def measure_inference_time(self):
        inference_times = {}

        for model_name, model in self.models.items():
            start_time = time.time()
            
            if isinstance(model, KNeighborsClassifier):  # sklearn model
                model.fit(self.X_train, self.y_train)
                model.predict(self.X_test)
            else:  # Custom model
                model.fit(self.X_train, self.y_train)
                model.predict(self.X_test)
            
            end_time = time.time()
            inference_times[model_name] = end_time - start_time

        self.inference_times = inference_times
        return inference_times

    def plot_inference_times(self):
        if not hasattr(self, 'inference_times'):
            raise ValueError("Inference times have not been measured yet. Call 'measure_inference_time' first.")
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.inference_times.keys(), self.inference_times.values(), color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Inference Time (seconds)')
        plt.title('Inference Time for Different KNN Models')
        plt.show()

    def measure_inference_time_vs_train_size(self, train_sizes):
        inference_times_vs_size = {}

        for model_name, model in self.models.items():
            times_for_model = []
            
            for size in train_sizes:
                X_train_subset = self.X_train[:int(size * len(self.X_train))]
                y_train_subset = self.y_train[:int(size * len(self.y_train))]
                
                start_time = time.time()
                model.fit(X_train_subset, y_train_subset)
                model.predict(self.X_test)
                end_time = time.time()
                
                times_for_model.append(end_time - start_time)
            
            inference_times_vs_size[model_name] = times_for_model

        self.inference_times_vs_size = inference_times_vs_size
        return inference_times_vs_size

    def plot_inference_time_vs_train_size(self, train_sizes):
        if not hasattr(self, 'inference_times_vs_size'):
            raise ValueError("Inference times vs train size have not been measured yet. Call 'measure_inference_time_vs_train_size' first.")
        
        plt.figure(figsize=(10, 6))

        for model_name, times in self.inference_times_vs_size.items():
            plt.plot(train_sizes, times, label=model_name)

        plt.xlabel('Train Dataset Size (Fraction)')
        plt.ylabel('Inference Time (seconds)')
        plt.title('Inference Time vs Train Dataset Size')
        plt.legend()
        plt.show()

        

