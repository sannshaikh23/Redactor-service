import cv2
import numpy as np

def _nms(boxes, overlap=0.3):
    """Non-maximum suppression for (x,y,w,h) boxes."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    pick = []

    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)

        xx1 = np.maximum(x1[last], x1[idxs[:-1]])
        yy1 = np.maximum(y1[last], y1[idxs[:-1]])
        xx2 = np.minimum(x2[last], x2[idxs[:-1]])
        yy2 = np.minimum(y2[last], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[last] + areas[idxs[:-1]] - inter)

        idxs = idxs[np.where(iou <= overlap)]
    return [tuple(map(int, boxes[i])) for i in pick]


class Redactor:
    def __init__(self):
        # Face cascades
        self.frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

        # License plate cascade (often weak outside RU plates, so we add a contour fallback)
        self.plate = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

        # CLAHE for contrast improvement
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _preprocess_gray(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = self.clahe.apply(gray)
        return gray

    def _detect_faces(self, gray):
        h, w = gray.shape[:2]
        # Dynamic minSize helps on mixed resolutions
        min_side = max(24, int(min(h, w) * 0.06))

        faces1 = self.frontal.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(min_side, min_side)
        )
        faces2 = self.profile.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(min_side, min_side)
        )
        boxes = list(faces1) + list(faces2)
        return _nms(boxes, overlap=0.25)

    def _detect_plates_haar(self, gray):
        if self.plate.empty():
            return []
        h, w = gray.shape[:2]
        min_w = max(40, int(w * 0.08))
        min_h = max(15, int(h * 0.03))
        plates = self.plate.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(min_w, min_h)
        )
        return list(plates)

    def _detect_plates_morph(self, gray):
        """
        Fallback plate detector using gradients + morphology + contour filtering.
        Works reasonably across regions when Haar misses.
        """
        img_h, img_w = gray.shape[:2]
        img_area = img_h * img_w

        # Denoise but keep edges
        blur = cv2.bilateralFilter(gray, 11, 17, 17)

        # Horizontal gradients highlight plate text
        gradx = cv2.Sobel(blur, cv2.CV_16S, 1, 0, ksize=3)
        gradx = cv2.convertScaleAbs(gradx)

        # Threshold & close gaps between characters
        _, bw = cv2.threshold(gradx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        morph = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.dilate(morph, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 0.0005 * img_area or area > 0.15 * img_area:
                continue
            ar = w / float(h + 1e-6)
            if ar < 2.0 or ar > 7.5:  # typical plate aspect
                continue
            # extent: fill ratio
            rect = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(rect, [c - [x, y]], -1, 255, thickness=cv2.FILLED)
            extent = rect.mean() / 255.0
            if extent < 0.35:
                continue
            boxes.append((x, y, w, h))

        return _nms(boxes, overlap=0.3)

    def _best_scale_for_detection(self, img):
        # Upscale small images to help Haar; cap to x2 to avoid artifacts
        h, w = img.shape[:2]
        max_dim = max(h, w)
        if max_dim < 800:
            scale = min(2.0, 800.0 / max_dim)
            det_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            return det_img, scale
        return img, 1.0

    def _dynamic_blur_kernel(self, w, h):
        k = int(max(w, h) * 0.25)
        k = k + 1 if k % 2 == 0 else k  # make odd
        return max(31, min(k, 151))

    def add_watermark(self, image, text="Redacted Image"):
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.0
        color = (0, 0, 255)   # red
        thickness = 2
        (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
        x = max(10, image.shape[1] - tw - 20)
        y = max(th + 10, image.shape[0] - 20)
        cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return image

    def redact(self, input_path, output_path, mode="blur"):
        """
        Returns True if something was redacted, False if no faces/plates detected.
        On False, no output file is written to avoid confusion.
        """
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Invalid image")

        # Work on a detection-friendly scale
        det_img, scale = self._best_scale_for_detection(image)
        gray = self._preprocess_gray(det_img)

        # --- DETECT ---
        faces = self._detect_faces(gray)

        # Plates: Haar + morphology fallback
        plates = self._detect_plates_haar(gray)
        plates_fallback = self._detect_plates_morph(gray)
        all_plates = _nms(list(plates) + list(plates_fallback), overlap=0.3)

        # Map detections back to original scale if we upscaled
        def _map_back(boxes):
            if scale == 1.0:
                return boxes
            mapped = []
            for (x, y, w, h) in boxes:
                mapped.append((int(x / scale), int(y / scale), int(w / scale), int(h / scale)))
            return mapped

        faces = _map_back(faces)
        all_plates = _map_back(all_plates)

        detections = faces + all_plates
        if len(detections) == 0:
            # Nothing found: do NOT write an output image (prevents "unblurred redacted_*.jpg")
            return False

        # --- REDACT ---
        out = image.copy()
        for (x, y, w, h) in detections:
            x = max(0, x); y = max(0, y)
            w = max(1, w); h = max(1, h)
            x2 = min(out.shape[1], x + w)
            y2 = min(out.shape[0], y + h)

            roi = out[y:y2, x:x2]
            if mode == "fill":
                roi[:] = (0, 0, 0)
            else:
                k = self._dynamic_blur_kernel(w, h)
                roi = cv2.GaussianBlur(roi, (k, k), 0)
            out[y:y2, x:x2] = roi

        out = self.add_watermark(out, "Redacted Image")
        cv2.imwrite(output_path, out)
        return True
