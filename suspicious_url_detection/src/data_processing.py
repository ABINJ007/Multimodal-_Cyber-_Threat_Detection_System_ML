import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data():
    # Replace the filename with the correct path to your CSV file
    df = pd.read_csv('data/malicious_phish.csv')  # update path if needed
    df.columns = [col.strip().lower() for col in df.columns]  # normalize column names
    return df

def prepare_data(df):
    # Filter benign URLs only
    df_benign = df[df['type'] == 'benign'].copy()
    df_benign['label'] = 0  # All benign labeled as 0

    # Optional: limit to reduce memory usage
    df_benign = df_benign.sample(n=50000, random_state=42)

    urls = df_benign['url'].values

    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(urls)

    return X, vectorizer
