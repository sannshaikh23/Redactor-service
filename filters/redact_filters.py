import cv2
import numpy as np

class RedactFilter:
    def __init__(self, method="blur"):
        """
        RedactFilter supports two methods:
        - "blur": Gaussian blur inside the region
        - "fill": Black fill inside the region
        """
        if method not in ["blur", "fill"]:
            raise ValueError("Invalid method. Use 'blur' or 'fill'.")
        self.method = method

    def apply(self, image, boxes):
        """
        Apply the redaction filter to the given image.

        Args:
            image (np.ndarray): The input image (BGR).
            boxes (list of tuples): List of bounding boxes (x, y, w, h).

        Returns:
            np.ndarray: The redacted image.
        """
        out = image.copy()
        for (x, y, w, h) in boxes:
            x, y = max(0, x), max(0, y)
            x2, y2 = min(out.shape[1], x + w), min(out.shape[0], y + h)

            roi = out[y:y2, x:x2]

            if self.method == "fill":
                roi[:] = (0, 0, 0)
            else:  # blur
                k = max(15, int(min(w, h) * 0.3))
                k = k + 1 if k % 2 == 0 else k  # kernel must be odd
                roi = cv2.GaussianBlur(roi, (k, k), 0)
                out[y:y2, x:x2] = roi

        return out
