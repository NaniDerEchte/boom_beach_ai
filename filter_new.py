import os
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np
import shutil
from multiprocessing import Pool, cpu_count
import psutil

def preprocess_reference_frames(reference_dir):
    reference_frames = []
    for file in os.listdir(reference_dir):
        if file.endswith('.jpg'):
            reference_frame_path = os.path.join(reference_dir, file)
            reference_frame = cv2.imread(reference_frame_path)
            if reference_frame is not None:
                reference_frames.append(reference_frame)
    return reference_frames

def is_gameplay_frame(frame, reference_frames, threshold=0.2):
    for reference_frame in reference_frames:
        min_height = min(frame.shape[0], reference_frame.shape[0])
        min_width = min(frame.shape[1], reference_frame.shape[1])
        frame_resized = cv2.resize(frame, (min_width, min_height))
        reference_frame_resized = cv2.resize(reference_frame, (min_width, min_height))
        
        # Dynamische Anpassung der Fenstergröße
        win_size = min(7, min(frame_resized.shape[:2]) - 1)
        win_size = max(3, win_size)
        if win_size % 2 == 0:
            win_size -= 1
        
        try:
            score = ssim(frame_resized, reference_frame_resized, 
                         win_size=win_size, 
                         channel_axis=-1,  # Für Farbbilder
                         data_range=255)
            if score < threshold:
                return True
        except ValueError as e:
            print(f"Fehler beim Vergleich der Bilder: {e}")
            continue
    return False

def process_frame(args):
    frame_path, reference_frames, threshold, output_dir, non_gameplay_dir = args
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Konnte Bild nicht laden: {frame_path}")
        return
    try:
        if is_gameplay_frame(frame, reference_frames, threshold):
            relative_path = os.path.relpath(frame_path, start=input_dir)
            output_frame_path = os.path.join(output_dir, relative_path)
            os.makedirs(os.path.dirname(output_frame_path), exist_ok=True)
            shutil.move(frame_path, output_frame_path)
        else:
            relative_path = os.path.relpath(frame_path, start=input_dir)
            non_gameplay_frame_path = os.path.join(non_gameplay_dir, relative_path)
            os.makedirs(os.path.dirname(non_gameplay_frame_path), exist_ok=True)
            shutil.move(frame_path, non_gameplay_frame_path)
        print(f"Processed frame {frame_path}")
    except Exception as e:
        print(f"Fehler bei der Verarbeitung von {frame_path}: {e}")


def filter_gameplay_frames(input_dir, output_dir, non_gameplay_dir, reference_dir, threshold=0.2):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(non_gameplay_dir):
        os.makedirs(non_gameplay_dir)
    
    reference_frames = preprocess_reference_frames(reference_dir)
    if not reference_frames:
        print("Keine Referenzbilder gefunden. Bitte überprüfen Sie das Referenzverzeichnis.")
        return
    
    frame_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.jpg'):
                frame_path = os.path.join(root, file)
                frame_paths.append((frame_path, reference_frames, threshold, output_dir, non_gameplay_dir))
    
    # Berechne die optimale Anzahl von Prozessen basierend auf CPU und RAM
    max_processes = min(cpu_count(), int(psutil.virtual_memory().available / (psutil.virtual_memory().total * 0.2)))
    max_processes = max(1, int(max_processes * 0.90))  # Nutze maximal 90% der verfügbaren Ressourcen
    
    print(f"Verwende {max_processes} Prozesse")
    
    with Pool(processes=max_processes) as pool:
        for _ in pool.imap_unordered(process_frame, frame_paths):
            pass

if __name__ == "__main__":
    input_dir = '/home/nani/boom_beach_ai/frames/krabbe_bilder/'
    output_dir = '/home/nani/boom_beach_ai/frames/filtered_frames/'
    non_gameplay_dir = '/home/nani/boom_beach_ai/frames/non_gameplay_frames/'
    reference_dir = '/home/nani/boom_beach_ai/frames/reference_frame/'
    filter_gameplay_frames(input_dir, output_dir, non_gameplay_dir, reference_dir, threshold=0.2)
