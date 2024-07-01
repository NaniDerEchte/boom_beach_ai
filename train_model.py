# Version 6 (Stable)

import torch
import torch.multiprocessing as mp
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import time
import re
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# Definieren Sie einen Ordner für die Zwischenspeicherung des Modells
checkpoint_dir = '/home/nani/boom_beach_ai/checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

class BoomBeachDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        
        for subdir in sorted(os.listdir(data_dir)):
            subdir_path = os.path.join(data_dir, subdir)
            if os.path.isdir(subdir_path):
                sorted_images = sorted(
                    [os.path.join(subdir_path, file) for file in os.listdir(subdir_path) if file.endswith('.jpg')],
                    key=lambda x: int(re.search(r'(\d+)\.jpg$', x).group(1))
                )
                self.image_files.extend(sorted_images)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Definieren Sie die Transformationen
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Erstellen Sie den Dataset- und DataLoader
data_dir = '/home/nani/boom_beach_ai/frames/filtered_frames/'
dataset = BoomBeachDataset(data_dir, transform=transform)
num_workers = mp.cpu_count()
batch_size = 64  # Erhöhen Sie dies, wenn Ihr Speicher es zulässt
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# Definieren Sie das Modell
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 10)  # Angenommen, wir haben 10 Klassen

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)

# Verlustfunktion und Optimierer definieren
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Funktion zum Speichern des Fortschritts
def save_checkpoint(epoch, batch_idx, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_batch_{batch_idx}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint gespeichert: {checkpoint_path}")

# Funktion zum Laden des neuesten Checkpoints
def load_latest_checkpoint():
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
    if not checkpoints:
        return None, 0, 0, float('inf')
    
    latest_checkpoint = max(checkpoints, key=lambda x: (int(x.split('_')[2]), int(x.split('_')[4].split('.')[0])))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')  # Entfernen Sie mmap=True
    end_time = time.time()
    print(f"Checkpoint Ladezeit: {end_time - start_time:.4f} Sekunden")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_idx = checkpoint['batch_idx']
    loss = checkpoint['loss']
    
    print(f"Neuester Checkpoint geladen: {checkpoint_path}")
    return model, epoch, batch_idx, loss

# Laden des neuesten Checkpoints, falls vorhanden
loaded_model, start_epoch, start_batch, best_loss = load_latest_checkpoint()
if loaded_model is not None:
    model = loaded_model
    print(f"Training wird von Epoche {start_epoch + 1}, Batch {start_batch + 1} fortgesetzt")
else:
    print("Training startet von Anfang an")
    start_epoch = 0
    start_batch = 0

# Training des Modells
num_epochs = 10
save_interval = 100  # Speichern alle 100 Batches

for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch_idx, (images, img_paths) in enumerate(progress_bar):
        if epoch == start_epoch and batch_idx < start_batch:
            continue
        
        images = images.to(device)
        labels = torch.zeros(images.size(0), dtype=torch.long).to(device)  # Dummy-Labels
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Aktualisieren Sie die Fortschrittsanzeige
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        # Speichern des Fortschritts
        if (batch_idx + 1) % save_interval == 0:
            save_checkpoint(epoch, batch_idx, model, optimizer, loss.item())
    
    avg_loss = running_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    # Speichern des Checkpoints am Ende jeder Epoche
    save_checkpoint(epoch, len(dataloader)-1, model, optimizer, avg_loss)
    
    scheduler.step()

# Speichern des finalen Modells
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Finales Modell gespeichert als '{final_model_path}'")

