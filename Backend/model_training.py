import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


malicious_df = pd.read_csv(r'C:\Users\abinj\ABXx\VS Code\AI_ Based_Multimodel_Threat_Detection\Backend\Dataset\malicious_phish.csv')
malicious_df = malicious_df[['url', 'type']]
malicious_df['label'] = malicious_df['type'].apply(lambda x: 0 if x == 'benign' else 1)
malicious_df = malicious_df[['url', 'label']]


url_class_df = pd.read_csv(
    r'C:\Users\abinj\ABXx\VS Code\AI_ Based_Multimodel_Threat_Detection\Backend\Dataset\URL Classification.csv',
    header=None,  
    names=['id', 'url', 'type']  
)
url_class_df = url_class_df[['id', 'url', 'type']]
url_class_df = url_class_df.rename(columns={'URL': 'url', 'Type': 'type'})
print('heelo')

malicious_types = ['Adult', 'Malware', 'Phishing']  
url_class_df['label'] = url_class_df['type'].apply(lambda x: 1 if x in malicious_types else 0)
print('heelo')

url_class_df = url_class_df[['url', 'label']]
print('heelo')

combined_df = pd.concat([malicious_df, url_class_df], ignore_index=True)
combined_df.dropna(subset=['url'], inplace=True)
combined_df.drop_duplicates(subset=['url'], inplace=True)
print('heelo')

X_train, X_test, y_train, y_test = train_test_split(
    combined_df['url'], combined_df['label'], test_size=0.3, random_state=42
)
print('heelo')

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 5))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print('heelo')

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_vec, y_train)


predictions = clf.predict(X_test_vec)
print(classification_report(y_test, predictions))


joblib.dump(clf, "url_threat_model.pkl")
joblib.dump(vectorizer, "url_vectorizer.pkl")

print("✅ Model training completed and saved.")
