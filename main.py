from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from rembg import remove
import torch.nn as nn
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

## CODE FOR MILESTONE 1 DATA AUG ##
# base_path = "asl_alphabet_train/"
# imgs = []
# for x in range(1, 101):  # Load 100 images
#     imgs.append(base_path + 'A/A' + str(x) + '.jpg')

# # Create a figure and subplots
# plt.figure(figsize=(20, 20))  # Adjust the figure size to fit 100 images

# # Define transforms
# transforms = v2.Compose([
#     v2.ToTensor(),  # Converts PIL image to PyTorch tensor (C, H, W)
#     v2.RandomHorizontalFlip(p=0.5),
#     v2.RandomVerticalFlip(p=0.5),
# ])

# # Loop through the first 100 images for the letter A and add them to the subplots
# for idx, image_path in enumerate(imgs):
#     # Load the image using PIL
#     img = Image.open(image_path)
#     img = remove(img)
#     img = transforms(img)  # Apply transforms (converts to tensor)

#     # Convert tensor from (C, H, W) to (H, W, C) for matplotlib
#     img = img.permute(1, 2, 0)  # Permute dimensions
#     # Create a subplot in a 10x10 grid
#     plt1 = plt.subplot(10, 10, idx + 1)  # 10x10 grid for 100 images
#     plt1.imshow(img)
#     plt1.set_title(f'{idx + 1}') 
#     plt1.axis('off')  # Hide axes for better visualization

# plt.tight_layout()  # Adjust spacing between subplots
# plt.show()

dir_path ='Dataset/asl_alphabet_train'

### DATA LOADER CLASS ###
class ASLDataset(Dataset):

    def __init__(self,root_dir,transform=None):
        
        ## DATA PROCESSIONG ##
        self.root_dir = root_dir
        self.transform = transform

        # List class names (each subdirectory is assumed to be a class)
        self.classes = os.listdir(root_dir)
            ## Code from: https://pynative.com/python-list-files-in-a-directory/#:~:text=sample.txt'%5D-,os.,all%20of%20its%20subdirectories%2Fsubfolders.
        
        # Create a mapping from class name to a numeric label
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Lists to hold image file paths and corresponding labels
        self.directory = {}
        
        # Loop over each class folder to make a list of all the files
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.directory[os.path.join(class_dir, file)]= self.class_to_idx[class_name]


    def dataAugment(self,noOfFiles=1,idx=1):
        img, _ = self.__getitem__(idx)
        x = Image.open(ASLDataset[idx][0])
        for i in range(noOfFiles):
            y = transforms(x)
            y.save(f"Dataset/asl_alphabet_train/DA{i}.jpg")

        transforms = v2.Compose([
            v2.ToTensor(),  # Converts PIL image to PyTorch tensor (C, H, W)
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])

    def __len__(self):
        return len(self.directory)
    
    def __getitem__(self,idx):
        return list(self.directory)[idx]
        
        
### MODEL CLASS ###
class ASLCNN(nn.Module):
    def __init__(self, num_classes):
        super(ASLCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)  # Downsample the feature maps
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Assuming input images are 224x224
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))

        x = x.view(x.size(0), -1)  # Flattening
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x
    
### TRAINING FUNCTION ###
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

### TESTING FUNCTION ###
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


train_dataset = ASLDataset("Dataset/asl_alphabet_train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_classes = len(train_dataset.classes)
model = ASLCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

### TRAINING THE MODEL ###
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

### TESTING THE MODEL ###
test_dataset = ASLDataset("Dataset/asl_alphabet_test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_model(model, test_loader)