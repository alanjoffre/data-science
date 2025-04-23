# face_recognition_module.py

import cv2
import face_recognition
from db_connection import load_encodings
from concurrent.futures import ThreadPoolExecutor

class FaceRecognition:
    def __init__(self):
        """Inicializa a classe FaceRecognition e carrega rostos conhecidos."""
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

    def load_known_faces(self):
        """Carrega rostos conhecidos do banco de dados."""
        self.known_face_encodings = []
        self.known_face_names = []

        encodings = load_encodings()
        for name, encoding in encodings:
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)

    def live_recognition(self):
        """Realiza o reconhecimento facial ao vivo."""
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erro: Não foi possível acessar a câmera.")
            return

        print("Reconhecimento ao vivo iniciado. Pressione 'q' para sair.")

        with ThreadPoolExecutor() as executor:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Não foi possível ler o quadro da câmera.")
                    break

                future = executor.submit(self.process_frame, frame)
                frame = future.result()
                
                cv2.imshow('Face Recognition', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        video_capture.release()
        cv2.destroyAllWindows()

    def process_frame(self, frame):
        """Processa um quadro de vídeo para reconhecimento facial.

        Args:
            frame (ndarray): O quadro de vídeo a ser processado.

        Returns:
            ndarray: O quadro de vídeo processado com as detecções de rosto.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"

            if True in matches:
                first_match_index = matches.index(True)
                name = self.known_face_names[first_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 1)

        return frame
