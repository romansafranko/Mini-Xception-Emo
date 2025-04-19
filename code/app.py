"""
Real‑time emotion‑recognition desktop application.

Loads a pretrained MiniXception CNN and starts a PyQt5 GUI that
captures webcam frames, detects faces with OpenCV Haar cascades,
and overlays the predicted emotion label on the live video feed.
"""

import sys
import cv2
import torch
import torchvision.transforms as T
from PyQt5 import QtWidgets, QtGui, QtCore
from model import MiniXception

# ------------------------------------------------------------------
# Model‑checkpoint path (can be overridden from the command line)
# ------------------------------------------------------------------
default_model_path = '../results/best_model.pt'
model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path

# ------------------------------------------------------------------
# Model setup
# ------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create network skeleton and load weights
model = MiniXception(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()                                   # inference mode ‑ no gradients

# ------------------------------------------------------------------
# Pre‑processing transform applied to a *single* face crop
# ------------------------------------------------------------------
transform = T.Compose([
    T.Grayscale(),            # model was trained on grayscale images
    T.Resize((128, 128)),     # network’s fixed input resolution
    T.ToTensor(),             # PIL → tensor
    T.Normalize([0.5], [0.5]) # map pixel range to approximately [‑1, 1]
])

# Ordered class labels (must match training time)
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------------------------------------------
# Inference helper
# ------------------------------------------------------------------
def predict_emotion(frame):
    """
    Detect the **first** face in *frame* (BGR) and return its predicted
    emotion as a string.  Returns ``None`` when no face is found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Fast but coarse Viola–Jones detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Crop + adapt to network input
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = T.ToPILImage()(face)
        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            out   = model(face)
            _, ix = torch.max(out, 1)
        return classes[ix.item()]              # return on first detected face
    return None                                # no face detected

# ------------------------------------------------------------------
# Qt5 GUI
# ------------------------------------------------------------------
class EmotionApp(QtWidgets.QWidget):
    """Minimal QWidget that streams webcam frames with emotion overlay."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real‑time Emotion Recognition")

        # Open default webcam (device 0)
        self.video = cv2.VideoCapture(0)

        # QLabel used as a canvas for the rendered QPixmap
        self.label = QtWidgets.QLabel(self)
        self.label.setFixedSize(640, 480)

        # Timer triggers *update_frame* roughly every 30 ms
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        """
        Pull one frame from the webcam, annotate it, convert BGR → RGB,
        wrap it in a QImage and show it inside the QLabel.
        """
        ret, frame = self.video.read()
        if not ret:
            return

        emotion = predict_emotion(frame)
        if emotion:
            cv2.putText(frame, emotion, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch  = frame.shape
        stride    = ch * w                         # bytes per line
        q_img     = QtGui.QImage(frame.data, w, h, stride,
                                 QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(q_img))

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = EmotionApp()
    window.show()
    app.exec_()
