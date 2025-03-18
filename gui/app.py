from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QLabel, QPushButton, QVBoxLayout
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import QThread
import sys
from os import remove
from camera import CameraApp

class StartWindow(QWidget):
    def __init__(self, parent=None):
        super(StartWindow, self).__init__(parent)
        self.setWindowTitle("Simple ASL Decoder - Start")
        self.setGeometry(500, 200, 750, 750)

        # Create and set up the camera widget.
        self.camera_widget = CameraApp()
        self.camera_widget.graphicsView.setParent(self)
        self.camera_widget.graphicsView.setGeometry(125, 100, 500, 300)

        # Create a button to trigger reading the sign.
        self.readButton = QPushButton("Read Sign!", self)
        self.readButton.setGeometry(300, 500, 150, 50)  # x, y, width, height


class ResultsWindow(QWidget):
    def __init__(self, parent=None):
        super(ResultsWindow, self).__init__(parent)
        # Use a layout to add the label (setCentralWidget is not available in QWidget)
        layout = QVBoxLayout(self)
        self.label = QLabel(self)
        pixmap = QPixmap('temp_image_in.png')
        self.label.setPixmap(pixmap)
        layout.addWidget(self.label)
        self.setLayout(layout)

    def updatePhoto(self, filename="temp_image_in.png"):
        pixmap = QPixmap(filename)
        self.label.setPixmap(pixmap)

class MainApp(QMainWindow):
    def __init__(self):
        super(MainApp, self).__init__()
        self.setWindowTitle("Simple ASL Decoder")

        # Create a stacked widget and set it as the central widget.
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        # Create instances of StartWindow and ResultsWindow.
        self.start = StartWindow()
        self.result = ResultsWindow()

        # Add the windows to the stack.
        self.stack.addWidget(self.start)
        self.stack.addWidget(self.result)

        # Connect the read button signal to switch to the results view.
        self.start.readButton.clicked.connect(self.takePhotoAndShowResults)
        
    def takePhotoAndShowResults(self):
        # Capture the photo using the CameraApp's capturePhoto method.
        self.start.camera_widget.capturePhoto()
        # Update the results window with the captured image.
        self.result.updatePhoto()
        # Switch to the results view.
        self.stack.setCurrentWidget(self.result)

    def closeEvent(self, event):
        print("Closing the application...")
        remove('temp_image_in.png')
        event.accept()

        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())
