from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as v2
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
        self.classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
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


    def dataAugment(self, noOfFiles=1, idx=1):
        file_path = list(self.directory.keys())[idx]
        from torchvision.io import read_image
        original_img = read_image(file_path).float() / 255.0  
        augment_transform = v2.Compose([
            v2.Resize((64, 64)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
        ])
        to_pil = v2.ToPILImage()
        for i in range(noOfFiles):
            augmented = augment_transform(original_img)
            augmented_img = to_pil(augmented)
            augmented_img.save(f"Dataset/asl_alphabet_train/DA_{i}.jpg")

    def __len__(self):
        return len(self.directory)
    
    def __getitem__(self, idx):
        file_path = list(self.directory.keys())[idx]
        from torchvision.io import read_image  
        image = read_image(file_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        label = self.directory[file_path]
        return image, label
        
        
### MODEL CLASS ###
class ASLCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ASLCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
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
### TRAINING LOOP ###
import torch.optim as optim

#  Define parameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Define transforms for dataset (using v2 from torchvision.transforms)
transform = v2.Compose([
    v2.Resize((64, 64)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

# Create dataset and dataloader
dataset = ASLDataset(root_dir=dir_path, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

num_classes = len(dataset.classes)
model = ASLCNN(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

### TESTING LOOP ###
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader: 
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")