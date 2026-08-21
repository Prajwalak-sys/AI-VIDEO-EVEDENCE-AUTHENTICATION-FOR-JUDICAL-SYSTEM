import tensorflow as tf
import numpy as np
import os

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)


def load_model():

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    return model


def extract_features(model, image_path):

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE
    )

    image_array = tf.keras.utils.img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = preprocess_input(image_array)

    features = model.predict(
        image_array,
        verbose=0
    )

    return features[0]


def process_all_frames(model, frames_folder):

    features = []
    frame_names = []

    for filename in sorted(os.listdir(frames_folder)):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            image_path = os.path.join(
                frames_folder,
                filename
            )

            frame_features = extract_features(
                model,
                image_path
            )

            features.append(frame_features)
            frame_names.append(filename)

            print(f"Processed: {filename}")

    return np.array(features), frame_names


if __name__ == "__main__":

    print("Loading MobileNetV2...")

    model = load_model()

    print("MobileNetV2 loaded successfully!")

    frames_folder = "processed_frames"

    features, frame_names = process_all_frames(
        model,
        frames_folder
    )

    print()
    print("Feature extraction completed!")
    print(f"Total frames processed: {len(frame_names)}")
    print(f"Feature array shape: {features.shape}")

    np.save(
        "spatial_features.npy",
        features
    )

    np.save(
        "frame_names.npy",
        np.array(frame_names)
    )

    print("Features saved to spatial_features.npy")
    print("Frame names saved to frame_names.npy")