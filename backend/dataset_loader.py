import os


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def load_dataset(dataset_folder):

    videos = []
    labels = []

    real_folder = os.path.join(
        dataset_folder,
        "real"
    )

    fake_folder = os.path.join(
        dataset_folder,
        "fake"
    )

    # Load real videos
    if os.path.exists(real_folder):

        for filename in sorted(os.listdir(real_folder)):

            if filename.lower().endswith(VIDEO_EXTENSIONS):

                video_path = os.path.join(
                    real_folder,
                    filename
                )

                videos.append(video_path)
                labels.append(0)

    # Load fake videos
    if os.path.exists(fake_folder):

        for filename in sorted(os.listdir(fake_folder)):

            if filename.lower().endswith(VIDEO_EXTENSIONS):

                video_path = os.path.join(
                    fake_folder,
                    filename
                )

                videos.append(video_path)
                labels.append(1)

    return videos, labels


if __name__ == "__main__":

    dataset_folder = "dataset"

    videos, labels = load_dataset(
        dataset_folder
    )

    print(f"Total videos: {len(videos)}")

    print(f"Real videos: {labels.count(0)}")

    print(f"Fake videos: {labels.count(1)}")

    print()

    for video, label in zip(videos, labels):

        label_name = "REAL" if label == 0 else "FAKE"

        print(
            f"{label_name}: {video}"
        )