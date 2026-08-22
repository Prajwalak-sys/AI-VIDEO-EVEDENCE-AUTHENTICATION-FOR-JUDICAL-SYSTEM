import cv2
import numpy as np
import os


def calculate_optical_flow(previous_frame, current_frame):

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_frame,
        cv2.COLOR_BGR2GRAY
    )

    flow = cv2.calcOpticalFlowFarneback(
        previous_gray,
        current_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )

    magnitude, angle = cv2.cartToPolar(
        flow[..., 0],
        flow[..., 1]
    )

    motion_score = np.mean(magnitude)

    return float(motion_score)


def process_frames(frames_folder):

    frame_files = []

    for filename in sorted(
        os.listdir(frames_folder)
    ):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            frame_files.append(filename)

    scores = []
    frame_names = []

    if len(frame_files) < 2:

        print("Not enough frames for optical flow.")

        return (
            np.array(scores),
            frame_names
        )

    previous_path = os.path.join(
        frames_folder,
        frame_files[0]
    )

    previous_frame = cv2.imread(
        previous_path
    )

    for filename in frame_files[1:]:

        current_path = os.path.join(
            frames_folder,
            filename
        )

        current_frame = cv2.imread(
            current_path
        )

        if previous_frame is None or current_frame is None:

            previous_frame = current_frame

            continue

        score = calculate_optical_flow(
            previous_frame,
            current_frame
        )

        scores.append(score)

        frame_names.append(filename)

        print(
            f"{filename}: "
            f"{score:.4f}"
        )

        previous_frame = current_frame

    return (
        np.array(scores),
        frame_names
    )


if __name__ == "__main__":

    frames_folder = "processed_frames"

    print(
        "Starting Optical Flow analysis..."
    )

    scores, frame_names = process_frames(
        frames_folder
    )

    print()

    print(
        "Optical Flow analysis completed!"
    )

    print(
        f"Total frame transitions analyzed: "
        f"{len(frame_names)}"
    )

    if len(scores) > 0:

        print(
            f"Average motion score: "
            f"{np.mean(scores):.4f}"
        )

        print(
            f"Maximum motion score: "
            f"{np.max(scores):.4f}"
        )

        print(
            f"Minimum motion score: "
            f"{np.min(scores):.4f}"
        )

    np.save(
        "optical_flow_scores.npy",
        scores
    )

    np.save(
        "optical_flow_frame_names.npy",
        np.array(frame_names)
    )

    print()

    print(
        "Optical Flow scores saved to "
        "optical_flow_scores.npy"
    )

    print(
        "Frame names saved to "
        "optical_flow_frame_names.npy"
    )