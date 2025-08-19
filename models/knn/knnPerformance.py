import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

class KNNPerformanceEvaluator:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
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


