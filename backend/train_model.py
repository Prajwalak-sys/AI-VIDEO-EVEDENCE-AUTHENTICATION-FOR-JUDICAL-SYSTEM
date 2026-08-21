import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


FEATURES_FILE = "training_features.npy"
LABELS_FILE = "training_labels.npy"


def load_training_data():

    if not os.path.exists(FEATURES_FILE):
        print("training_features.npy not found.")
        print("Run dataset_preparation.py first.")
        return None, None

    if not os.path.exists(LABELS_FILE):
        print("training_labels.npy not found.")
        print("Run dataset_preparation.py first.")
        return None, None

    X = np.load(FEATURES_FILE)
    y = np.load(LABELS_FILE)

    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    return X, y


def build_model(sequence_length):

    model = Sequential([
        Input(
            shape=(sequence_length, 1280)
        ),

        Bidirectional(
            LSTM(
                64,
                return_sequences=False
            )
        ),

        Dropout(0.3),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model():

    X, y = load_training_data()

    if X is None:
        return

    if len(X) < 2:
        print("Not enough videos for training.")
        return

    sequence_length = X.shape[1]

    model = build_model(
        sequence_length
    )

    print()
    print("BiLSTM model created.")
    print()

    model.summary()

    print()
    print("Starting training...")

    model.fit(
        X,
        y,
        epochs=10,
        batch_size=4,
        validation_split=0.2,
        shuffle=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        "models/zero_bilstm.keras"
    )

    print()
    print("Training completed!")
    print(
        "Model saved to "
        "models/zero_bilstm.keras"
    )


if __name__ == "__main__":

    train_model()