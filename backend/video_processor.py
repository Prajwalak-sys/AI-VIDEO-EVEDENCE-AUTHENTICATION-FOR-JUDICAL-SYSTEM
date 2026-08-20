import cv2
import os


def extract_frames(video_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    video = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:

        success, frame = video.read()

        if not success:
            break

        frame_path = os.path.join(
            output_folder,
            f"frame_{frame_count:05d}.jpg"
        )

        cv2.imwrite(frame_path, frame)

        frame_count += 1

    video.release()

    return frame_count


def get_video_info(video_path):

    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video.release()

    duration = total_frames / fps if fps > 0 else 0

    return fps, total_frames, width, height, duration


if __name__ == "__main__":

    video_path = "videos/WhatsApp Video 2026-08-20 at 3.15.57 PM.mp4"
    output_folder = "frames"

    total_frames = extract_frames(video_path, output_folder)

    print(f"Total frames extracted: {total_frames}")

    fps, total, width, height, duration = get_video_info(video_path)

    print(f"FPS: {fps}")
    print(f"Total frames: {total}")
    print(f"Resolution: {width} x {height}")
    print(f"Duration: {duration:.2f} seconds")