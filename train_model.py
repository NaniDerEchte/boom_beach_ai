import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import time
import re
import torch.nn.functional as F

# Definieren Sie einen Ordner für die Zwischenspeicherung des Modells
checkpoint_dir = '/home/nani/boom_beach_ai/checkpoints/'
os.makedirs(checkpoint_dir, exist_ok=True)

class BoomBeachDataset(Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir

        self.transform = transform

        self.image_files = []

        for root, _, files in os.walk(data_dir):

            for file in files:

                if file.endswith('.jpg'):

                    # Extrahieren Sie die Nummer am Ende des Dateinamens

                    match = re.search(r'(\d+)\.jpg$', file)

                    if match:

                        frame_number = int(match.group(1))

                        self.image_files.append((os.path.join(root, file), frame_number))

                    else:

                        # Wenn keine Nummer gefunden wurde, fügen Sie die Datei ans Ende

                        self.image_files.append((os.path.join(root, file), float('inf')))

        

        # Sortieren Sie die Liste basierend auf den extrahierten Nummern

        self.image_files.sort(key=lambda x: x[1])

        self.image_files = [file[0] for file in self.image_files]  # Behalten Sie nur die Dateipfade



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
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Definieren Sie das Modell (vereinfachtes Beispiel)
import torch.nn as nn

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

model = SimpleCNN()

# Verlustfunktion und Optimierer definieren
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Funktion zum Speichern des Fortschritts
def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint gespeichert: {checkpoint_path}")

# Training des Modells
num_epochs = 10
save_interval = 100  # Speichern alle 100 Batches

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, (images, img_paths) in enumerate(dataloader):
        start_time = time.time()
        
        optimizer.zero_grad()
        outputs = model(images)
        # Hier würden Sie normalerweise die echten Labels verwenden
        # Für dieses Beispiel verwenden wir Dummy-Labels
        loss = criterion(outputs, torch.zeros(images.size(0), dtype=torch.long))
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        end_time = time.time()
        batch_time = end_time - start_time
        
        # Ausgabe für jedes verarbeitete Bild
        for i, img_path in enumerate(img_paths):
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], '
                  f'Image: {img_path}, Time: {batch_time/len(img_paths):.4f} seconds')
        
        # Speichern des Fortschritts
        if (batch_idx + 1) % save_interval == 0:
            save_checkpoint(epoch, model, optimizer, loss.item())
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')

# Speichern des finalen Modells
final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
torch.save(model.state_dict(), final_model_path)
print(f"Finales Modell gespeichert als '{final_model_path}'")
