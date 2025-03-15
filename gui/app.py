# import sys
# from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QWidget
# from camera import CameraApp
# from PyQt6.QtCore import pyqtSlot

# This works, need to convert into a class and add a button to capture the image and save it to a file and add a close function
# def main():
#    # Create the application instance
#     app = QApplication(sys.argv)

#    # Create the main window
#     window = QMainWindow()
#     window.setWindowTitle("Simple ASL Decoder")
#     window.setGeometry(500, 200, 750, 750) #starting position x, y and width, height

#     camera_widget = CameraApp()
#     camera_widget.graphicsView.setParent(window)
#     camera_widget.graphicsView.setGeometry(125, 100, 500, 300)

#     button = QPushButton("Hello World!", window)
#     button.clicked.connect(lambda: camera_widget.capturePhoto())
#     button.setGeometry(300, 500, 150, 50) # x, y, width, height
#     window.show()
#     sys.exit(app.exec())


from PyQt6.QtWidgets import * 
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import sys 
from os import remove
from camera import CameraApp
class MainApp(QWidget): 
    def __init__(self): 
        super(MainApp, self).__init__() 

        self.start = QWidget()
        self.results = QWidget()
        # Define GUI

        self.startPage()
        # self.resultPage()

        self.Stack = QStackedWidget(self)

        self.Stack.addWidget(self.start)
        self.Stack.addWidget(self.results)


        self.show() 
    
    def startPage(self):
        self.setWindowTitle("Simple ASL Decoder - Start")
        self.setGeometry(500, 200, 750, 750)

        camera_widget = CameraApp()
        camera_widget.graphicsView.setParent(self)
        camera_widget.graphicsView.setGeometry(125, 100, 500, 300)

        button = QPushButton("Read Sign!", self)
        button.clicked.connect(lambda: camera_widget.capturePhoto())
        button.clicked.connect(lambda: self.resultPage())
        button.setGeometry(300, 500, 150, 50) # x, y, width, height

    def resultPage(self):
        self.setWindowTitle("Simple ASL Decoder - Results")
        self.setGeometry(500, 200, 750, 750)


    def closeEvent(self, event):
        # Handle the close event
        print("Closing the application...")
        remove('temp_image_in.png')
        event.accept()

# start the app 
def main():
    app = QApplication(sys.argv)
    window = MainApp()
    sys.exit(app.exec()) 

if __name__ == '__main__':
    main()