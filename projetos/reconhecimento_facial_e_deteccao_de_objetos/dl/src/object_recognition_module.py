import cv2
import numpy as np

class ObjectRecognition:
    def __init__(self):
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.layer_names = self.net.getLayerNames()
        self.output_layers = self.get_output_layers()
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def get_output_layers(self):
        """Obtém as camadas de saída da rede YOLO."""
        try:
            layer_names = self.net.getLayerNames()
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            return output_layers
        except IndexError as e:
            print(f"Erro ao obter camadas de saída: {e}")
            return []

    def object_recognition(self):
        """Realiza o reconhecimento de objetos ao vivo usando YOLO."""
        if not self.output_layers:
            print("Erro: As camadas de saída não foram carregadas corretamente.")
            return

        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Erro: Não foi possível acessar a câmera.")
            return

        print("Reconhecimento de objetos iniciado. Pressione 'q' para sair.")

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Não foi possível ler o quadro da câmera.")
                break

            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('Object Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
