import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def run_inference(model, X_train, X_test, y_test):
    # Predict probabilities or logits
    y_pred_probs = model.predict(X_test)  # shape: (num_samples, 1) for sigmoid
    
    # Threshold probabilities at 0.5 to get class predictions (binary classification)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    
    # If y_test is one-hot encoded, convert to label indices
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test.flatten()
    
    # Print confusion matrix and classification report
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
