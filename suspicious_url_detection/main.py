from src.data_processing import load_data, prepare_data
from suspicious_url_detection.src.autoencoder_model import build_autoencoder

def main():
    print("🔁 Loading dataset...")
    df = load_data()

    print("📐 Preprocessing and vectorizing benign URLs only...")
    X_train, vectorizer = prepare_data(df)

    print("🧠 Building and training the autoencoder...")
    model = build_autoencoder(input_dim=X_train.shape[1])

    # Convert to dense matrix after reducing feature size
    X_train_dense = X_train.toarray()

    model.fit(
        X_train_dense,
        X_train_dense,
        epochs=10,
        batch_size=64,
        shuffle=True,
        validation_split=0.1
    )

    # Save model, vectorizer, etc. (optional)

if __name__ == "__main__":
    main()
