import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load malicious phishing dataset
malicious_df = pd.read_csv('Dataset/malicious_phish.csv')
malicious_df = malicious_df[['url', 'type']]


malicious_df['label'] = malicious_df['type'].apply(lambda x: 0 if x == 'benign' else 1)
malicious_df = malicious_df[['url', 'label']]

malicious_df.dropna(subset=['url'], inplace=True)
malicious_df.drop_duplicates(subset=['url'], inplace=True)


X_train, X_test, y_train, y_test = train_test_split(
    malicious_df['url'], malicious_df['label'], test_size=0.3, random_state=42
)

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)


predictions = clf.predict(X_test_vec)
print(classification_report(y_test, predictions))


joblib.dump(clf, "url_threat_model.pkl")
joblib.dump(vectorizer, "url_vectorizer.pkl")

print("✅ Model training completed and saved.")
