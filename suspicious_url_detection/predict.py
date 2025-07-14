from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def prepare_data(df, benign_label='benign'):
    df['label'] = df['type'].apply(lambda x: 0 if str(x).lower() == benign_label.lower() else 1)
    
    # Split URL text and labels
    X = df['url']
    y = df['label'].values

    # Split before fitting vectorizer
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Apply TF-IDF vectorizer
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=10000)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    return X_train, X_test, y_train, y_test, vectorizer
