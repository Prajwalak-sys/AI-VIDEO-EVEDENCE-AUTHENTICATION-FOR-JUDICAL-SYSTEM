import cv2
import numpy as np
import os


def calculate_dct_score(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = np.float32(gray)

    dct = cv2.dct(gray)

    magnitude = np.abs(dct)

    high_frequency = magnitude[1:, 1:]

    dct_score = np.mean(
        high_frequency
    )

    return float(dct_score)


def process_all_frames(
    frames_folder,
    output_folder
):

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    scores = []
    frame_names = []

    for filename in sorted(
        os.listdir(frames_folder)
    ):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                frames_folder,
                filename
            )

            image = cv2.imread(
                image_path
            )

            if image is None:
                continue

            score = calculate_dct_score(
                image
            )

            scores.append(score)
            frame_names.append(filename)

            print(
                f"{filename}: "
                f"{score:.4f}"
            )

    scores = np.array(
        scores,
        dtype=np.float32
    )

    np.save(
        os.path.join(
            output_folder,
            "dct_scores.npy"
        ),
        scores
    )

    np.save(
        os.path.join(
            output_folder,
            "dct_frame_names.npy"
        ),
        np.array(frame_names)
    )

    return scores, frame_names


if __name__ == "__main__":

    frames_folder = "processed_frames"

    output_folder = "analysis"

    print(
        "Starting DCT analysis..."
    )

    scores, frame_names = process_all_frames(
        frames_folder,
        output_folder
    )

    print()

    print(
        "DCT analysis completed!"
    )

    print(
        f"Total frames analyzed: "
        f"{len(frame_names)}"
    )

    if len(scores) > 0:

        print(
            f"Average DCT score: "
            f"{np.mean(scores):.4f}"
        )

        print(
            f"Maximum DCT score: "
            f"{np.max(scores):.4f}"
        )

        print(
            f"Minimum DCT score: "
            f"{np.min(scores):.4f}"
        )

    print()

    print(
        f"DCT results saved to: "
        f"{output_folder}"
    )