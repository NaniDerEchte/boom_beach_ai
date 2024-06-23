# extract_frames.py
import cv2
import os

def extract_frames(video_dir='videos', frame_dir='frames'):
    os.makedirs(frame_dir, exist_ok=True)
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        frame_subdir = os.path.join(frame_dir, video_name)
        os.makedirs(frame_subdir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(frame_subdir, f"{video_name}_frame_{count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            count += 1
        cap.release()

if __name__ == "__main__":
    extract_frames(video_dir='/home/nani/boom_beach_ai/videos/', frame_dir='/home/nani/boom_beach_ai/frames/krabbe_bilder/')
