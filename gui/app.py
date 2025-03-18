from PyQt6.QtWidgets import * 
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys 
from os import remove
from camera import CameraApp

class startWindow(QWidget):
    def __init__(self, parent=None):
        super(startWindow, self).__init__(parent)
        self.setWindowTitle("Simple ASL Decoder - Start")
        self.setGeometry(500, 200, 750, 750)

        camera_widget = CameraApp()
        camera_widget.graphicsView.setParent(self)
        camera_widget.graphicsView.setGeometry(125, 100, 500, 300)

        readButton = QPushButton("Read Sign!", self)

        readButton.setGeometry(300, 500, 150, 50) # x, y, width, height


class ResultsWindow(QWidget): 
    def __init__(self, parent=None): 
        super(ResultsWindow, self).__init__(parent)
        
        label = QLabel(self)
        pixmap = QPixmap('temp_image_in.png')
        label.setPixmap(pixmap)
        self.setCentralWidget(label)


class MainApp(QMainWindow): 
    def __init__(self): 
        super(MainApp, self).__init__() 
        self.setWindowTitle("Simple ASL Decoder")

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.start = startWindow()
        self.result = ResultsWindow()

        self.stack.addWidget(self.start)
        self.stack.addWidget(self.result)

        self.startWindow.readButton.clicked.connect(self.showResults)
        

    def showResults(self):
        self.stack.setCurrentWidget(self.start)

    # def closeEvent(self, event):
    #     # Handle the close event
    #     print("Closing the application...")
    #     remove('temp_image_in.png')
    #     event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec())