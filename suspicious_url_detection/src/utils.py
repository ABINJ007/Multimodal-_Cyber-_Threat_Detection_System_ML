# src/utils.py

import numpy as np
from sklearn.metrics import classification_report

def compute_threshold(model, X_train):
    reconstructions = model.predict(X_train)
    mse = np.mean(np.power(X_train - reconstructions, 2), axis=1)
    threshold = np.mean(mse) + 3 * np.std(mse)
    return threshold

def predict_suspicious(model, X, threshold):
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)
    return mse > threshold, mse

def evaluate(y_true, y_pred, target_label='Benign'):
    y_true_binary = [0 if label.lower() == target_label.lower() else 1 for label in y_true]
    print(classification_report(y_true_binary, y_pred, target_names=['Benign', 'Suspicious']))
