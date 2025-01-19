from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from face_recognition_module import FaceRecognition
from object_recognition_module import ObjectRecognition
from user_management_module import UserManagementWindow
from auth_module import authenticate
import sys

class FaceRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        if authenticate():
            self.face_recognition = FaceRecognition()
            self.object_recognition = ObjectRecognition()
            self.initUI()
        else:
            sys.exit()

    def initUI(self):
        self.setWindowTitle("Face and Object Recognition App")
        self.setGeometry(100, 100, 400, 300)

        # Botões e rótulos
        self.live_recognition_button = QPushButton("Iniciar Reconhecimento ao Vivo", self)
        self.live_recognition_button.clicked.connect(self.face_recognition.live_recognition)

        self.register_button = QPushButton("Cadastro de Usuário", self)
        self.register_button.clicked.connect(self.open_user_management)

        self.object_recognition_button = QPushButton("Reconhecimento de Objetos", self)
        self.object_recognition_button.clicked.connect(self.object_recognition.object_recognition)

        self.stop_button = QPushButton("Fechar Aplicação", self)
        self.stop_button.clicked.connect(self.close_app)

        self.label = QLabel("Pressione um botão para começar.", self)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.register_button)
        layout.addWidget(self.live_recognition_button)
        layout.addWidget(self.object_recognition_button)
        layout.addWidget(self.stop_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def open_user_management(self):
        """Abre o gerenciamento de usuários"""
        self.user_management_window = UserManagementWindow(self)
        self.user_management_window.exec_()

    def close_app(self):
        """Fecha a aplicação"""
        self.close()
