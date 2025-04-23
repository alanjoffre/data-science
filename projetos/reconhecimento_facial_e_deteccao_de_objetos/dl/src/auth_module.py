from PyQt5.QtWidgets import QDialog, QLineEdit, QLabel, QPushButton, QVBoxLayout, QMessageBox

class AuthDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Autenticação")
        self.setGeometry(100, 100, 300, 150)

        self.label = QLabel("Digite sua senha:", self)
        self.password_input = QLineEdit(self)
        self.password_input.setEchoMode(QLineEdit.Password)

        self.login_button = QPushButton("Login", self)
        self.login_button.clicked.connect(self.check_password)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.login_button)

        self.setLayout(layout)

    def check_password(self):
        password = self.password_input.text()
        if password == "12345678":  # Defina uma senha segura aqui
            self.accept()
        else:
            QMessageBox.warning(self, "Erro", "Senha incorreta!")

def authenticate(password):
    """Verifica se a senha está correta."""
    if password == "12345678":  # Defina uma senha segura aqui
        return True
    return False

