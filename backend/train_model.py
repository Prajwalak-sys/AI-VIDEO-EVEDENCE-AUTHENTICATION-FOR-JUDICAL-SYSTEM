import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

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

    if len(X) < 4:
        print("Not enough videos for train/test split.")
        print("Add more labeled videos first.")
        return

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print()
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    sequence_length = X.shape[1]

    model = build_model(
        sequence_length
    )

    print()
    print("BiLSTM model created successfully.")

    print()
    model.summary()

    print()
    print("Starting training...")

    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=4,
        validation_split=0.2,
        shuffle=True
    )

    print()
    print("Evaluating model on unseen test videos...")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=1
    )

    print()
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    os.makedirs(
        "models",
        exist_ok=True
    )

    model.save(
        "models/zero_bilstm.keras"
    )

    print()
    print("Model saved successfully!")
    print(
        "Location: models/zero_bilstm.keras"
    )


if __name__ == "__main__":

    train_model()