# import sys
# from PySide6.QtWidgets import QApplication, QPushButton, QMainWindow
# from PySide6.QtCore import Slot

# @Slot()
# def read_sign():
#     print("Button Clicked")


# def main():
#     app = QApplication(sys.argv)

#     window = QMainWindow()
#     window.setWindowTitle("Hello World")
#     window.setGeometry(100, 100, 280, 80)

#     button = QPushButton("Hello World!")
#     button.clicked.connect(read_sign)
#     button.move(150, 80)
#     button.show()
    
#     window.show()

#     sys.exit(app.exec())

# if __name__ == "__main__":
#     main()

import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from camera import CameraApp


def main():
   # Create the application instance
    app = QApplication(sys.argv)

   # Create the main window
    window = QMainWindow()
    window.setWindowTitle("Simple PyQt Example")
    window.setGeometry(500, 200, 750, 750) #starting position x, y and width, height

    camera_widget = CameraApp()
    camera_widget.graphicsView.setParent(window)
    camera_widget.graphicsView.setGeometry(125, 100, 500, 300)


#    # Create a label widget
#    label = QLabel("Hello, PyQt!", window)
#    label.move(150, 80)

    button = QPushButton("Hello World!", window)
    button.setGeometry(300, 500, 150, 50) # x, y, width, height
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
   main()

