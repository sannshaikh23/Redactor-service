import cv2
import os

class OpenCVDetector:
    def __init__(self):
        # Face cascade (always included in OpenCV)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
            raise FileNotFoundError("Face cascade not found in OpenCV!")

        # License plate cascade (optional)
        plate_cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_russian_plate_number.xml')
        if os.path.exists(plate_cascade_path):
            self.plate_cascade = cv2.CascadeClassifier(plate_cascade_path)
        else:
            print(f"Warning: License plate cascade '{plate_cascade_path}' not found. Plate detection disabled.")
            self.plate_cascade = None

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces

    def detect_plates(self, image):
        if self.plate_cascade is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, 1.1, 4)
        return plates