# CMPM17-Final-ASLDecoder

This project is a work-in-progress final assignment for the CMPM 17 class. It involves using a Convolutional Neural Network (CNN) for American Sign Language (ASL) recognition. The goal is to build an ASL decoder that can process live video input and classify hand signs using a trained deep learning model.

## 🚧 Project Status: In Progress 🚧
The project is actively being developed in `app.py`. The authors are currently working on:
- Implementing live video capture for real-time ASL detection.
- Improving the CNN model for better accuracy.
- Integrating the trained model with a PyQt6-based GUI.

## 📂 Project Structure
📦 CMPM17-Final-ASLDecoder 
┣ 📂 Dataset/ 
┃  ┗ 📂 asl_alphabet_train/ # ASL training dataset 
┣ 📜 app.py # Main application file (WIP) 
┣ 📜 model.py # CNN model and training script 
┣ 📜 requirements.txt # Dependencies 
┗ 📜 README.md # Project documentation

## 🛠 Dependencies
To run this project, you need the following dependencies:
- Python 3.x
- PyQt6
- PySide6
- Torch & Torchvision
- NumPy
- OpenCV

### Dataset:
We are using the ASL Alphabet dataset from Kaggle:  
[Download the dataset here](https://www.kaggle.com/datasets/grassknoted/asl-alphabet?resource=download)

## 🔧 Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/CMPM17-Final-ASLDecoder.git
   cd CMPM17-Final-ASLDecoder
   ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```
3. Download and extract the dataset inside the Dataset/ folder.


4. Run the application:

```bash
python app.py
```

## 🖥 Current Features
- CNN model for ASL recognition.
- Data augmentation for improved training.
- Real-time ASL detection from video input (In Progress).
- GUI for user interaction (Planned).