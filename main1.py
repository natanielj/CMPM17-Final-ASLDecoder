from torchvision import transforms as v2
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import os

# Directory for dataset
dir_path = 'Dataset/asl_alphabet_train'

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Only consider directories (classes)
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        # Build lists of image paths and labels
        self.image_paths, self.labels = [], []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[class_name])

    def dataAugment(self, noOfFiles=1, idx=0):
        # Load image as tensor and normalize to [0, 1]
        original_img = read_image(self.image_paths[idx]).float() / 255.0
        augment_transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        # Convert tensor to PIL image for saving
        to_pil = v2.ToPILImage()
        for i in range(noOfFiles):
            augmented = augment_transform(original_img)
            augmented_img = to_pil(augmented)
            augmented_img.save(f"Dataset/asl_alphabet_train/DA_{i}.jpg")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image as tensor and normalize to [0, 1]
        image = read_image(self.image_paths[idx]).float() / 255.0
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

class ASLCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        # For 64x64 inputs, four poolings reduce spatial size to 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Transformation for tensor images (no need for ToTensor)
transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

dataset = ASLDataset(root_dir=dir_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.classes)
model = ASLCNN(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Testing loop
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")