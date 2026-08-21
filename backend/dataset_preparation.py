import os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


IMAGE_SIZE = (224, 224)

# Number of frames used from each video
SEQUENCE_LENGTH = 32


VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


def load_model():

    model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        pooling="avg"
    )

    return model


def get_sampled_frames(video_path):

    video = cv2.VideoCapture(video_path)

    total_frames = int(
        video.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        video.release()
        return []

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        SEQUENCE_LENGTH,
        dtype=int
    )

    frames = []

    for index in frame_indices:

        video.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index)
        )

        success, frame = video.read()

        if success:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                IMAGE_SIZE
            )

            frames.append(frame)

    video.release()

    return frames


def extract_video_features(model, video_path):

    frames = get_sampled_frames(
        video_path
    )

    if len(frames) != SEQUENCE_LENGTH:

        print(
            f"Skipping {video_path} - "
            f"only {len(frames)} frames available"
        )

        return None

    frames = np.array(
        frames,
        dtype=np.float32
    )

    frames = preprocess_input(
        frames
    )

    features = model.predict(
        frames,
        verbose=0
    )

    return features


def load_videos_from_folder(
    folder_path,
    label
):

    videos = []

    if not os.path.exists(folder_path):

        return videos

    for filename in sorted(
        os.listdir(folder_path)
    ):

        if filename.lower().endswith(
            VIDEO_EXTENSIONS
        ):

            video_path = os.path.join(
                folder_path,
                filename
            )

            videos.append(
                (video_path, label)
            )

    return videos


def prepare_dataset():

    real_videos = load_videos_from_folder(
        "dataset/real",
        0
    )

    fake_videos = load_videos_from_folder(
        "dataset/fake",
        1
    )

    all_videos = (
        real_videos +
        fake_videos
    )

    print(
        f"Real videos found: "
        f"{len(real_videos)}"
    )

    print(
        f"Fake videos found: "
        f"{len(fake_videos)}"
    )

    print(
        f"Total videos found: "
        f"{len(all_videos)}"
    )

    if len(all_videos) == 0:

        print()
        print(
            "No training videos found."
        )

        return

    print()
    print(
        "Loading MobileNetV2..."
    )

    model = load_model()

    print(
        "MobileNetV2 loaded!"
    )

    dataset_features = []
    dataset_labels = []

    for video_path, label in all_videos:

        print()
        print(
            f"Processing: {video_path}"
        )

        features = extract_video_features(
            model,
            video_path
        )

        if features is not None:

            dataset_features.append(
                features
            )

            dataset_labels.append(
                label
            )

            print(
                f"Feature shape: "
                f"{features.shape}"
            )

    if len(dataset_features) == 0:

        print(
            "No videos were successfully processed."
        )

        return

    X = np.array(
        dataset_features,
        dtype=np.float32
    )

    y = np.array(
        dataset_labels,
        dtype=np.int32
    )

    np.save(
        "training_features.npy",
        X
    )

    np.save(
        "training_labels.npy",
        y
    )

    print()
    print(
        "Dataset preparation completed!"
    )

    print(
        f"Training features shape: "
        f"{X.shape}"
    )

    print(
        f"Training labels shape: "
        f"{y.shape}"
    )

    print(
        "Saved: training_features.npy"
    )

    print(
        "Saved: training_labels.npy"
    )


if __name__ == "__main__":

    prepare_dataset()