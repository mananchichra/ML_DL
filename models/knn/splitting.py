import numpy as np
import pandas as pd

class StratifiedSplitter:
    def __init__(self, target_col, train_size=0.8, val_size=0.1, random_state=None):
        self.target_col = target_col
        self.train_size = train_size
        self.val_size = val_size
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)

    def fit_transform(self, X, y):
        if isinstance(y, pd.DataFrame):
            y = y[self.target_col]
        
        df = pd.concat([X, pd.Series(y, name=self.target_col)], axis=1)
        
        df = df.groupby(self.target_col, group_keys=False).apply(lambda x: x.sample(frac=1, random_state=self.random_state))
        
        train_list = []
        val_list = []
        test_list = []
        
        for _, group in df.groupby(self.target_col):
            n_train = int(self.train_size * len(group))
            n_val = int(self.val_size * len(group))
            
            train_list.append(group.iloc[:n_train])
            val_list.append(group.iloc[n_train:n_train+n_val])
            test_list.append(group.iloc[n_train+n_val:])
        
        train_df = pd.concat(train_list).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        val_df = pd.concat(val_list).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        test_df = pd.concat(test_list).sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        X_train = train_df.drop(self.target_col, axis=1)
        y_train = train_df[self.target_col]
        
        X_val = val_df.drop(self.target_col, axis=1)
        y_val = val_df[self.target_col]
        
        X_test = test_df.drop(self.target_col, axis=1)
        y_test = test_df[self.target_col]
        
        return X_train, X_val, X_test, y_train, y_val, y_test

