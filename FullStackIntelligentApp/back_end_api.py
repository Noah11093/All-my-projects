import numpy as np 
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random

import joblib

class BackEnd:
    
    def __init__(self):
        self.df = pd.read_csv("./StudentData_refined.csv")
        
    
    def exploration(self, df):
        # Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. 
        print('Descriptions: ')
        print(df.describe(), '\n')
        print('Types: ')
        print(df.info(), '\n')
        df_stats = df.describe()
        df_stats.loc['Range'] = df_stats.loc['max'] - df_stats.loc['min']
        print('Range:')
        print(df_stats.loc['Range'], '\n')
        
        # Statistical assessments including means, averages, correlations
        print('Means:')
        print(df_stats.loc['mean'], '\n')
        print('Median:')
        print(df_stats.loc['50%'], '\n')
        print('Correlation:')
        print(df.corr(), '\n')

        # Missing data evaluations
        print(df.shape)

        print('Missing Values')
        print(df.isnull().sum(), '\n')

        print('Number of ?')
        print(df.isin(["?"]).sum())

        print(df.shape)
        
        
    def preprocessing(self, df=None):
        if df is None:
            df = pd.read_csv("./StudentData_refined.csv")
        
        df_clean = df.drop_duplicates()
        df_clean.replace('?', np.nan, regex=False, inplace=True)
        df_clean.drop('School', axis=1, inplace=True)
        df_clean = df_clean._convert(numeric=True)
        
        # With high correlation: replaced by mean
        df_clean['First Term Gpa'].fillna(df_clean['First Term Gpa'].mean(), inplace=True)
        df_clean['Second Term Gpa'].fillna(df_clean['Second Term Gpa'].mean(), inplace=True)
        
        # With low correlation: replaced by mode
        df_clean['First Language'].fillna(df_clean['First Language'].mode()[0], inplace=True)
        
        # With few missing values: delete rows
        df_clean.dropna(subset=['Previous Education'], inplace=True)
        df_clean.dropna(subset=['Age Group'], inplace=True)
        
        # With many missing values but low correlation: delete columns
        df_clean.drop('High School Average Mark', axis=1, inplace=True)
        df_clean.drop('Math Score', axis=1, inplace=True)
        df_clean.drop('English Grade', axis=1, inplace=True)
        
        return df_clean
    
    
    def modelling(self):
        df_clean = self.preprocessing(self.df)
    
        X, y = df_clean.iloc[:, :-1], df_clean.iloc[:, -1]
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=24, stratify=y)
        X_train, X_val, y_train, y_val  = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=24, stratify=y_train_full)

        best_model = None
        best_accuracy = 0
        n_iterations = 10  # Number of iterations for random search

        for _ in range(n_iterations):
            # Randomly choose hyperparameters
            units1 = random.choice([50, 100, 200, 300])
            units2 = random.choice([50, 100, 150])
            dropout_rate = random.choice([0.1, 0.2, 0.3])
            learning_rate = random.choice([0.01, 0.001, 0.0001])

            model = tf.keras.models.Sequential([
                tf.keras.layers.Dense(units1, activation='relu'),
                tf.keras.layers.Dense(units2, activation='relu'),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(80, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(20, activation='relu'),
                tf.keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])

            model.fit(X_train, y_train, batch_size=32, validation_data=(X_val, y_val), epochs=10)  # Reduced epochs for faster iteration

            accuracy = model.evaluate(X_val, y_val)[1]
            print(f"Iteration accuracy: {accuracy}, Parameters: units1={units1}, units2={units2}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        # Save the best model
        joblib.dump(best_model, 'nn_model.joblib', compress=9)
    
        # Evaluate the best model
        print("Best model performance on test set:", best_model.evaluate(X_test, y_test)[1])
        
        print(best_model.summary())

        
test = BackEnd()
test.modelling()


