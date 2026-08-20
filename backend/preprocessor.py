import cv2
import os
import numpy as np


IMAGE_SIZE = (224, 224)


def preprocess_frame(frame_path, output_path):

    frame = cv2.imread(frame_path)

    if frame is None:
        raise ValueError(f"Could not read frame: {frame_path}")

    frame = cv2.resize(frame, IMAGE_SIZE)

    frame = frame.astype(np.float32) / 255.0

    processed_frame = (frame * 255).astype(np.uint8)

    cv2.imwrite(output_path, processed_frame)

    return output_path


def preprocess_frames(input_folder, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    processed_count = 0

    for filename in sorted(os.listdir(input_folder)):

        if filename.lower().endswith((".jpg", ".jpeg", ".png")):

            input_path = os.path.join(input_folder, filename)

            output_path = os.path.join(
                output_folder,
                filename
            )

            preprocess_frame(
                input_path,
                output_path
            )

            processed_count += 1

    return processed_count


if __name__ == "__main__":

    input_folder = "frames/WhatsApp Video 2026-08-20 at 3.15.57 PM"

    output_folder = "processed_frames"

    total_processed = preprocess_frames(
        input_folder,
        output_folder
    )

    print(f"Total frames preprocessed: {total_processed}")