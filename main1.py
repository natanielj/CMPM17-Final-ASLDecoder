from torchvision import transforms
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import os
import random

# Directory for dataset
dir_path = '/asl_alphabet_train'

# Class for dataset
class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        # Basic Definition
        self.root_dir = root_dir
        self.transform = transform

        # More details on how this code works in engineering notebook
        # Get's the list of all the folders under the main directory
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))] 
        #   Code is from:  https://www.geeksforgeeks.org/python-list-files-in-a-directory/

        #Enumerating the code
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Getting the main file paths and their labels
        self.image_paths, self.labels, self.char = [], [], []

        # Getting labels from file directory
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[class_name])
                    self.char.append(class_name)

        for i in range(0,len(self.image_paths)):

            if random.choice([True, False, False, False, False, False, False, False]):

            #Data Augmentation function definition
                original_img = read_image(self.image_paths[i]).float() / 255.0
                augment_transform = transforms.Compose([
                    transforms.Resize((64, 64)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                ])

                # Convert tensor to PIL image for saving
                to_pil = transforms.ToPILImage()
                for j in range(5):
                    augmented = augment_transform(original_img)
                    augmented_img = to_pil(augmented)
                    augmeneted_img_path = f"Dataset/asl_alphabet_train/{self.char[i]}/DA_{j}.jpg"
                    augmented_img.save(augmeneted_img_path)
                    self.image_paths.append(augmeneted_img_path)
                    self.labels.append(self.class_to_idx[self.char[i]])
                    self.char.append(self.char[i])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Read image as tensor and normalize to [0, 1]
        image = read_image(self.image_paths[idx]).float() / 255.0
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Class for CNN model defination
class ASLCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ASLCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

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

# Model instanciation
dataset = ASLDataset(root_dir=dir_path, transform=transforms)
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



#Filtering testing data
images = os.listdir("/asl_alphabet_test")
labels = []
for i in images:
    labels.append(i[0])

# Testing loop
model.eval()
correct, total = 0, 0
with torch.no_grad():
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")