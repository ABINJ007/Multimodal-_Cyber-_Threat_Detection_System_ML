import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv("malicious_phish.csv")  # Path to your dataset
print(f"✅ Dataset loaded with shape: {df.shape}")

# Filter only benign samples for autoencoder training
benign_df = df[df['type'] == 'benign']
print(f"✅ Benign samples: {benign_df.shape[0]}")

# Vectorize URLs using TF-IDF (reduced features to prevent memory issues)
vectorizer = TfidfVectorizer(max_features=1000)  # Reduced from 10000 to 1000
X = vectorizer.fit_transform(benign_df['url']).toarray()
print(f"✅ TF-IDF vectorized shape: {X.shape}")

# Labels (all 0 for benign)
y = np.zeros(X.shape[0])

# Split into validation set for threshold calculation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Save the validation data for training and threshold calculation
np.save("data/X_val.npy", X_val)
np.save("data/y_val.npy", y_val)

print("✅ Preprocessing complete. Files saved:")
print("- data/X_val.npy")
print("- data/y_val.npy")
