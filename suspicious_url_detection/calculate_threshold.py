import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import joblib  


df_val = pd.read_csv('data/malicious_phish.csv')


df_val['label'] = (df_val['type'] != 'benign').astype(int)
labels = df_val['label'].values


features = np.array([len(url) for url in df_val['url']])
features = features.reshape(-1, 1)  

scores = (features - features.min()) / (features.max() - features.min())
scores = scores.flatten()

best_threshold = 0.0
best_f1 = 0.0
thresholds = np.linspace(0, 1, 101)

for threshold in thresholds:
    preds = (scores >= threshold).astype(int)
    f1 = f1_score(labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"✅ Best Threshold: {best_threshold:.4f}, F1 Score: {best_f1:.4f}")


joblib.dump(best_threshold, 'models/optimal_threshold.pkl')
print("✅ Saved threshold to 'models/optimal_threshold.pkl'")
