import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

printf("hello:")
df = pd.read_csv("malicious_phish.csv")


urls = df['url']
labels = df['type'].apply(lambda x: 0 if x == 'benign' else 1)  

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(urls).toarray()


X_benign = X[labels == 0]
X_suspicious = X[labels == 1]

X_train, X_val = train_test_split(X_benign, test_size=0.2, random_state=42)

input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation="relu")(input_layer)
encoded = Dense(256, activation="relu")(encoded)
decoded = Dense(512, activation="relu")(encoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer=Adam(1e-3), loss="mse")


autoencoder.fit(X_train, X_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_val, X_val),
                verbose=1)

X_test_benign = X_val
X_test_suspicious = X_suspicious[:len(X_test_benign)] 

benign_pred = autoencoder.predict(X_test_benign)
suspicious_pred = autoencoder.predict(X_test_suspicious)

benign_errors = [mean_squared_error(x, y) for x, y in zip(X_test_benign, benign_pred)]
suspicious_errors = [mean_squared_error(x, y) for x, y in zip(X_test_suspicious, suspicious_pred)]


threshold = np.mean(benign_errors) + 2 * np.std(benign_errors)
print(f"\nThreshold: {threshold:.6f}")


y_true = [0]*len(benign_errors) + [1]*len(suspicious_errors)
all_errors = benign_errors + suspicious_errors
y_pred = [1 if e > threshold else 0 for e in all_errors]


print("\nAccuracy:", accuracy_score(y_true, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))


roc_score = roc_auc_score(y_true, all_errors)
print(f"ROC-AUC Score: {roc_score:.4f}")

fpr, tpr, _ = roc_curve(y_true, all_errors)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_score:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder for URL Threat Detection")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()
