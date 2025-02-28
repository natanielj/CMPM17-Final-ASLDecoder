from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from rembg import remove

base_path = "./asl_alphabet_train/asl_alphabet_train/"
imgs = []
for x in range(1, 101):  # Load 100 images
    imgs.append(base_path + 'A/A' + str(x) + '.jpg')

# Create a figure and subplots
plt.figure(figsize=(20, 20))  # Adjust the figure size to fit 100 images

# Define transforms
transforms = v2.Compose([
    v2.ToTensor(),  # Converts PIL image to PyTorch tensor (C, H, W)
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
])

# Loop through the first 100 images for the letter A and add them to the subplots
for idx, image_path in enumerate(imgs):
    # Load the image using PIL
    img = Image.open(image_path)
    img = remove(img)
    img = transforms(img)  # Apply transforms (converts to tensor)

    # Convert tensor from (C, H, W) to (H, W, C) for matplotlib
    img = img.permute(1, 2, 0)  # Permute dimensions
    # Create a subplot in a 10x10 grid
    plt1 = plt.subplot(10, 10, idx + 1)  # 10x10 grid for 100 images
    plt1.imshow(img)
    plt1.set_title(f'{idx + 1}') 
    plt1.axis('off')  # Hide axes for better visualization

plt.tight_layout()  # Adjust spacing between subplots
plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear1 = nn.Linear(100, 200)
        self.linear2 = nn.Linear(200, 300)
        self.linear3 = nn.
