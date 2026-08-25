import cv2
import numpy as np
import os


def calculate_ela(image):

    temp_path = "ela_temp.jpg"

    cv2.imwrite(
        temp_path,
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 90]
    )

    compressed = cv2.imread(temp_path)

    if compressed is None:

        if os.path.exists(temp_path):
            os.remove(temp_path)

        return 0.0

    difference = cv2.absdiff(
        image,
        compressed
    )

    ela_image = cv2.convertScaleAbs(
        difference,
        alpha=10
    )

    ela_score = np.mean(
        ela_image
    )

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return float(ela_score)


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

            score = calculate_ela(
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
            "ela_scores.npy"
        ),
        scores
    )

    np.save(
        os.path.join(
            output_folder,
            "ela_frame_names.npy"
        ),
        np.array(frame_names)
    )

    return scores, frame_names


if __name__ == "__main__":

    frames_folder = "processed_frames"

    output_folder = "analysis"

    print(
        "Starting ELA analysis..."
    )

    scores, frame_names = process_all_frames(
        frames_folder,
        output_folder
    )

    print()
    print(
        "ELA analysis completed!"
    )

    print(
        f"Total frames analyzed: "
        f"{len(frame_names)}"
    )

    if len(scores) > 0:

        print(
            f"Average ELA score: "
            f"{np.mean(scores):.4f}"
        )

        print(
            f"Maximum ELA score: "
            f"{np.max(scores):.4f}"
        )

        print(
            f"Minimum ELA score: "
            f"{np.min(scores):.4f}"
        )

    print()
    print(
        f"ELA results saved to: "
        f"{output_folder}"
    )