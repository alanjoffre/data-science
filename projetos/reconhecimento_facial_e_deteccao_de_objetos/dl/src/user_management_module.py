from db_connection import save_encoding, load_encodings, delete_user  # Importa as funções do banco de dados
import cv2
import face_recognition

class UserManagement:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Carrega rostos conhecidos do banco de dados"""
        self.known_face_encodings = []
        self.known_face_names = []

        encodings = load_encodings()
        for name, encoding in encodings:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

    def register_user(self, name):
        """Registra um novo usuário capturando uma imagem e armazenando no banco de dados"""
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erro: Não foi possível acessar a câmera.")
            return

        print("Capturando imagem... Olhe para a câmera.")
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erro: Não foi possível capturar o frame.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if face_encodings:
                encoding = face_encodings[0]  # Considera apenas o primeiro rosto detectado
                save_encoding(name, encoding)  # Salva no banco de dados
                print(f"Usuário {name} registrado com sucesso.")
                self.load_known_faces()  # Atualiza os rostos conhecidos
                break
            else:
                print("Nenhum rosto detectado. Por favor, ajuste a posição da câmera.")

        video_capture.release()

    def list_users(self):
        """Lista todos os usuários registrados no banco de dados"""
        encodings = load_encodings()
        return [user[0] for user in encodings]

    def update_user(self, old_name, new_name):
        """Permite alterar o nome de um usuário registrado"""
        encodings = load_encodings()
        for name, encoding in encodings:
            if name == old_name:
                delete_user(old_name)  # Exclui o antigo
                save_encoding(new_name, encoding)  # Registra o novo nome
                print(f"Nome de {old_name} alterado para {new_name}.")
                return

        print(f"Usuário {old_name} não encontrado.")

    def delete_user(self, name):
        """Exclui um usuário do banco de dados"""
        delete_user(name)  # Exclui o usuário
        print(f"Usuário {name} excluído com sucesso.")
