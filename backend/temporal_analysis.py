import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout


def load_features():

    features = np.load("spatial_features.npy")

    print(f"Loaded feature shape: {features.shape}")

    return features


def build_bilstm_model():

    model = Sequential([
        Input(shape=(None, 1280)),

        Bidirectional(
            LSTM(64, return_sequences=False)
        ),

        Dropout(0.3),

        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":

    print("Loading spatial features...")

    features = load_features()

    print("Building BiLSTM model...")

    model = build_bilstm_model()

    print("BiLSTM model created successfully!")

    print()
    model.summary()