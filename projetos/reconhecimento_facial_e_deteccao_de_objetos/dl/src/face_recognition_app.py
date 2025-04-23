# face_recognition_app.py

from PyQt5.QtWidgets import QApplication
import sys
from ui_module import FaceRecognitionApp

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionApp()
    window.show()
    sys.exit(app.exec_())
