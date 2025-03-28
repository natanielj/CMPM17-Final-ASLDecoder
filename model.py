from torchvision.transforms import  v2
from torchvision.io import read_image
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
import os
import random
from PIL import Image
from torchvision.transforms.functional import to_pil_image
# import wandb
# Directory for dataset
dir_path = './asl_alphabet_train'

# Class for dataset
class ASLDataset(Dataset):
    def __init__(self, transform=None):

        # Basic Definition
        self.root_dir = './asl_alphabet_train'
        self.transform = transform

        # More details on how this code works in engineering notebook
        # Get's the list of all the folders under the main directory
        self.classes = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))] 
        #   Code is from:  https://www.geeksforgeeks.org/python-list-files-in-a-directory/

        #Enumerating the code
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Getting the main file paths and their labels
        self.image_paths, self.labels, self.char = [], [], []

        # Getting labels from file directory
        for class_name in self.classes:
            class_dir = os.path.join(self. root_dir, class_name)
            for file in os.listdir(class_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, file))
                    self.labels.append(self.class_to_idx[class_name])
                    self.char.append(class_name)

        for i in range(0,(len(self.image_paths) // 2)):

            if random.choice([True, False, False, False, False, False, False, False]):

            #Data Augmentation function definition
                original_img = Image.open(self.image_paths[i]).convert('RGB')
                trip = v2.Compose([
                    # transforms.Resize(200, 200),
                    v2.ToImage(),
                    v2.RandomVerticalFlip(p=0.5),
                    v2.ToDtype(torch.float32, scale=True),
                    v2.Normalize(mean=[0.485, 0.465, 0.406], std=[0.229, 0.224, 0.225])
                ])

                # Convert tensor to PIL image for saving

                for j in range(2):
                    aug = trip(original_img)
                    augmented = to_pil_image(aug)
                    augmeneted_img_path = f"asl_alphabet_train/{self.char[i]}/DA_{j}.jpg"
                    augmented.save(augmeneted_img_path)
                    self.image_paths.append(augmeneted_img_path)
                    self.labels.append(self.class_to_idx[self.char[i]])
                    self.char.append(self.char[i])

    def __len__(self):
        return (len(self.image_paths) // 2)

    def __getitem__(self, idx):
    # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')

        # Define transformation pipeline (if needed)
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((200, 200)),  # Resize all images to a fixed size
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image = transform(image)  # Apply transformations
        label = self.labels[idx]

        return image, torch.tensor(label, dtype=torch.long)

# Class for CNN model defination

class ASLCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ASLCNN, self).__init__()

        # b2tf = v2.Compose([
        #     v2.ToTensor()
        # ])

        # input_channels = b2tf(input_channels)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        print("running...")
        print(f"{x.shape}")

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Model instanciation
dataset = ASLDataset()
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
num_classes = len(dataset.classes)
model = ASLCNN(num_classes=num_classes)
device = torch.device("cuda")
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
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "./aslmodel")


#Filtering testing data
images = os.listdir("./asl_alphabet_test")
real_images = []
labels = []
for i in images:
    labels.append(i[0])
    img_path = os.path.join(images, i)
    img = Image.open(img_path).convert('RGB') 
    z = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((200, 200)),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    real_images.append(z(img))

real_images = torch.stack(real_images)

# # Start a new wandb run to track this script.
# run = wandb.init(
#     # Set the wandb entity where your project will be logged (generally your team name).
#     entity="pysigns",
#     # Set the wandb project where this run will be logged.
#     project="asl-decoder",
#     # Track hyperparameters and run metadata.
#     config={
#         "learning_rate": 0.01,
#         "architecture": "CNN",
#         "dataset": "ASL Alphabet",
#         "epochs": 10,
#     },
# )

# # Simulate training.
# epochs = 10
# offset = random.random() / 5
# for epoch in range(2, epochs):
#     acc = 1 - 2**-epoch - random.random() / epoch - offset
#     loss = 2**-epoch + random.random() / epoch + offset

#     # Log metrics to wandb.
#     run.log({"acc": acc, "loss": loss})

# # Finish the run and upload any remaining data.
# run.finish()


# Testing loop
model.eval()
correct, total = 0, 0
with torch.no_grad():
    # images, labels = images.to(device), labels.to(device)
    outputs = model(real_images)
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print(f"Accuracy: {100 * correct / total:.2f}%")


sd = model.state_dict()
torch.save(model.state_dict(), "./aslmodel")
