from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def train_model(X_train, y_train=None, epochs=10, batch_size=64):
    import numpy as np

    if y_train is None:
        y_train = np.zeros((len(X_train),))  # shape = (samples,)
    
    elif len(y_train.shape) > 1 and y_train.shape[1] == 2:
        y_train = y_train[:, 1]  # use just one column (malicious class)

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history
