import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import shutil
import multiprocessing as mp

def is_gameplay_frame(frame, reference_frames, threshold=0.15):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for reference_frame in reference_frames:
        gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_frame, gray_reference, full=True)
        if score < threshold:
            return True
    return False

def load_reference_frames(reference_dir):
    reference_frames = []
    for file in os.listdir(reference_dir):
        if file.endswith('.jpg'):
            reference_frame_path = os.path.join(reference_dir, file)
            reference_frame = cv2.imread(reference_frame_path)
            reference_frames.append(reference_frame)
    return reference_frames

def process_frame(args):
    frame_path, reference_frames, threshold, input_dir, output_dir, non_gameplay_dir = args
    frame = cv2.imread(frame_path)
    if is_gameplay_frame(frame, reference_frames, threshold):
        output_subdir = os.path.join(output_dir, os.path.relpath(os.path.dirname(frame_path), input_dir))
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        output_frame_path = os.path.join(output_subdir, os.path.basename(frame_path))
        shutil.move(frame_path, output_frame_path)
    else:
        non_gameplay_subdir = os.path.join(non_gameplay_dir, os.path.relpath(os.path.dirname(frame_path), input_dir))
        if not os.path.exists(non_gameplay_subdir):
            os.makedirs(non_gameplay_subdir)
        non_gameplay_frame_path = os.path.join(non_gameplay_subdir, os.path.basename(frame_path))
        shutil.move(frame_path, non_gameplay_frame_path)
    return f"Processed frame {frame_path}"

def filter_gameplay_frames(input_dir, output_dir, non_gameplay_dir, reference_dir, threshold=0.15):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(non_gameplay_dir):
        os.makedirs(non_gameplay_dir)
    
    reference_frames = load_reference_frames(reference_dir)
    
    frame_paths = []
    for subdir, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                frame_paths.append(os.path.join(subdir, file))
    
    num_cores = 26  # Hier wird die Anzahl der Kerne auf 26 festgelegt
    pool = mp.Pool(num_cores)
    
    args = [(frame_path, reference_frames, threshold, input_dir, output_dir, non_gameplay_dir) for frame_path in frame_paths]
    
    for result in pool.imap_unordered(process_frame, args):
        print(result)
    
    pool.close()
    pool.join()

input_dir = '/home/nani/boom_beach_ai/frames/krabbe_bilder/'
output_dir = '/home/nani/boom_beach_ai/frames/filtered_frames/'
non_gameplay_dir = '/home/nani/boom_beach_ai/frames/non_gameplay_frames/'
reference_dir = '/home/nani/boom_beach_ai/frames/reference_frame/'

if __name__ == '__main__':
    filter_gameplay_frames(input_dir, output_dir, non_gameplay_dir, reference_dir, threshold=0.15)
