import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# ✅ Load preprocessed data (make sure you're running this from the project root)
X = np.load("data/X_val.npy") 

# ✅ Build the autoencoder
input_dim = X.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(512, activation='relu')(input_layer)
encoded = Dense(256, activation='relu')(encoded)
encoded = Dense(128, activation='relu')(encoded)

decoded = Dense(256, activation='relu')(encoded)
decoded = Dense(512, activation='relu')(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# ✅ Set up callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train the autoencoder
autoencoder.fit(
    X, X, 
    epochs=20, 
    batch_size=256, 
    shuffle=True, 
    validation_split=0.2, 
    callbacks=[early_stop]
)

# ✅ Save the model
os.makedirs("models", exist_ok=True)
autoencoder.save("models/final_model.h5")

print("✅ Autoencoder model trained and saved as 'models/final_model.h5'")
