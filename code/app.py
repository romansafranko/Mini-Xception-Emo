"""
Run‑time GUI that grabs webcam frames, detects a face and predicts its
emotion with a trained MiniXception network.
"""

import sys
import cv2
import torch
import torchvision.transforms as T
from PyQt5 import QtWidgets, QtGui, QtCore
from model import MiniXception

# --------------------------------------------------------------------------- #
# 1. Model loading                                                            #
# --------------------------------------------------------------------------- #
default_model_path = "../results/best_model.pt"
model_path = sys.argv[1] if len(sys.argv) > 1 else default_model_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MiniXception(num_classes=7).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()  # inference‑only

# --------------------------------------------------------------------------- #
# 2. Input pre‑processing                                                     #
# --------------------------------------------------------------------------- #
transform = T.Compose(
    [
        T.Grayscale(),          # network expects a single channel
        T.Resize((128, 128)),   # training resolution
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

# Class labels follow the FER‑2013 dataset order
CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# --------------------------------------------------------------------------- #
# 3. Frame‑level helper                                                       #
# --------------------------------------------------------------------------- #
def predict_emotion(frame):
    """
    Detect face(s) in an RGB frame and return the first predicted emotion.

    Parameters
    ----------
    frame : np.ndarray (H × W × 3, BGR)
        Raw frame captured by OpenCV.

    Returns
    -------
    str | None
        Predicted label or *None* if no face was found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar cascade for frontal‑face detection (ships with OpenCV)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:          # process the first detected face
        face = gray[y : y + h, x : x + w]
        face = cv2.resize(face, (128, 128))
        face = T.ToPILImage()(face)
        face = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(face)
            label_idx = torch.argmax(logits, dim=1).item()
        return CLASSES[label_idx]

    return None  # no face detected


# --------------------------------------------------------------------------- #
# 4. PyQt5 GUI wrapper                                                        #
# --------------------------------------------------------------------------- #
class EmotionApp(QtWidgets.QWidget):
    """Lightweight window that shows the live webcam stream with predictions."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real‑time Emotion Recognition")

        # Video capture: 0 = default webcam
        self.video = cv2.VideoCapture(0)

        # QLabel used as a canvas for the RGB frame
        self.label = QtWidgets.QLabel(self)
        self.label.setFixedSize(640, 480)

        # 30 ms timer ≈ 33 FPS
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # --------------------------- event handlers ---------------------------- #
    def update_frame(self):
        """Grab a frame, run inference, render result to the GUI label."""
        ret, frame = self.video.read()
        if not ret:
            return

        emotion = predict_emotion(frame)
        if emotion:
            cv2.putText(
                frame,
                emotion,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=2,
            )

        # Convert BGR → RGB → QImage → QPixmap
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        q_img = QtGui.QImage(frame.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(q_img))


# --------------------------------------------------------------------------- #
# 5. Entry point                                                              #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = EmotionApp()
    window.show()
    app.exec_()
