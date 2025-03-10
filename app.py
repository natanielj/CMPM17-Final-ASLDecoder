import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton
from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
from PyQt6.QtMultimediaWidgets import QVideoWidget


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ASL Decoder")

        button = QPushButton("Press Me!")   
        self.setFixedSize(QSize(400, 300))

        self.setMenuWidget(button)


app = QApplication(sys.argv)
# capture_session = QMediaCaptureSession()

window = MainWindow()
window.show()

app.exec()