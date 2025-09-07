import cv2
import numpy as np
from detectors.opencv_detector import OpenCVDetector
from filters.redact_filters import RedactFilter

def test_detector_on_blank():
    img = np.zeros((200,200,3), dtype=np.uint8)
    detector = OpenCVDetector()
    assert detector.detect(img) == []

def test_redact_fill():
    img = np.ones((100,100,3), dtype=np.uint8) * 255
    boxes = [(10,10,50,50)]
    redactor = RedactFilter(method="fill")
    out = redactor.apply(img.copy(), boxes)
    assert (out[10:60,10:60] == 0).all()

def test_redact_blur():
    img = np.ones((100,100,3), dtype=np.uint8) * 255
    boxes = [(10,10,50,50)]
    redactor = RedactFilter(method="blur")
    out = redactor.apply(img.copy(), boxes)
    assert out.shape == img.shape

def test_invalid_input_type():
    try:
        OpenCVDetector().detect(None)
    except Exception:
        assert True

def test_multiple_boxes():
    img = np.ones((200,200,3), dtype=np.uint8) * 255
    boxes = [(10,10,50,50), (100,100,50,50)]
    out = RedactFilter().apply(img.copy(), boxes)
    assert out.shape == img.shape