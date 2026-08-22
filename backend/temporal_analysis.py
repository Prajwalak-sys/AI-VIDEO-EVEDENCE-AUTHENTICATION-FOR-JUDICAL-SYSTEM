import os
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Bidirectional,
    LSTM,
    Dense,
    Dropout
)


FEATURES_FILE = "fused_features.npy"

FEATURES_PER_FRAME = 1283


def load_fused_features():

    if not os.path.exists(FEATURES_FILE):

        print(
            "fused_features.npy not found."
        )

        print(
            "Run feature_fusion.py first."
        )

        return None

    features = np.load(
        FEATURES_FILE
    )

    print(
        f"Loaded fused feature shape: "
        f"{features.shape}"
    )

    return features


def build_bilstm_model(
    sequence_length,
    feature_count
):

    model = Sequential([

        Input(
            shape=(
                sequence_length,
                feature_count
            )
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

        Dropout(0.2),

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


if __name__ == "__main__":

    print(
        "Loading fused features..."
    )

    features = load_fused_features()

    if features is None:

        raise SystemExit

    if features.ndim != 2:

        print(
            "Unexpected feature format."
        )

        print(
            "Expected: "
            "(frames, features)"
        )

        raise SystemExit

    sequence_length = features.shape[0]

    feature_count = features.shape[1]

    print(
        f"Sequence length: "
        f"{sequence_length}"
    )

    print(
        f"Features per frame: "
        f"{feature_count}"
    )

    if feature_count != FEATURES_PER_FRAME:

        print(
            "Warning: Expected "
            f"{FEATURES_PER_FRAME} features "
            f"but found {feature_count}."
        )

    # Add batch dimension
    sequence = np.expand_dims(
        features,
        axis=0
    )

    print(
        f"BiLSTM input shape: "
        f"{sequence.shape}"
    )

    print()
    print(
        "Building fusion-aware BiLSTM..."
    )

    model = build_bilstm_model(
        sequence_length,
        feature_count
    )

    print(
        "Fusion-aware BiLSTM created successfully!"
    )

    print()

    model.summary()