import os
import shutil

# Pfade zu den Ordnern
source_folder = '/home/nani/boom_beach_ai/frames/filtered_frames/Hasty Mega Crab 2024 ｜ Stages 1-150/'
destination_folder = '/home/nani/boom_beach_ai/frames/krabbe_bilder/Hasty Mega Crab 2024 ｜ Stages 1-150/'

# Erstellen des Zielordners, falls er nicht existiert
os.makedirs(destination_folder, exist_ok=True)

# Durchlaufen aller Dateien im Quellordner
for filename in os.listdir(source_folder):
    # Vollständiger Pfad zur Datei
    source_file = os.path.join(source_folder, filename)
    destination_file = os.path.join(destination_folder, filename)
    
    # Überprüfen, ob es sich um eine Datei handelt
    if os.path.isfile(source_file):
        # Verschieben der Datei
        shutil.move(source_file, destination_file)
        print(f'Verschoben: {source_file} -> {destination_file}')

print('Alle Bilder wurden verschoben.')
