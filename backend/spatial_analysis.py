import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


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
        target_size=(224, 224)
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


if __name__ == "__main__":

    print("Loading MobileNetV2...")

    model = load_model()

    print("MobileNetV2 loaded successfully!")

    image_path = "processed_frames/frame_00000.jpg"

    features = extract_features(
        model,
        image_path
    )

    print(f"Feature shape: {features.shape}")

    print("First 10 features:")

    print(features[:10])