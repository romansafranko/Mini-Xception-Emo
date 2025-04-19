import sys
import cv2
import torch
import torchvision.transforms as T
from PyQt5 import QtWidgets, QtGui, QtCore
from model import MiniXception

default_model_path = '../results/best_model.pt'

model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path

# Načítanie modelu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MiniXception(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Transformácie pre vstup
transform = T.Compose([
    T.Grayscale(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5], [0.5])
])

# Triedy emócií
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Funkcia na predikciu emócie
def predict_emotion(frame):
    # Predspracovanie obrazu
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detekcia tváre (používam Haar cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (128, 128))
        face = T.ToPILImage()(face)
        face = transform(face).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(face)
            _, pred = torch.max(output, 1)
            emotion = classes[pred.item()]
        return emotion
    return None

# GUI aplikácia
class EmotionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Emotion Recognition")
        self.video = cv2.VideoCapture(0)
        self.label = QtWidgets.QLabel(self)
        self.label.setFixedSize(640, 480)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.video.read()
        if ret:
            emotion = predict_emotion(frame)
            if emotion:
                cv2.putText(frame, emotion, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.label.setPixmap(QtGui.QPixmap.fromImage(q_img))

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = EmotionApp()
    window.show()
    app.exec_()
