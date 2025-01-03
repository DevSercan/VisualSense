import numpy as np
import cv2
import time

class ColorDetector:
    def __init__(self, rgbColor: tuple, tolerance: int = 30, saveResult: bool = False):
        self.rgbColor = rgbColor
        self.tolerance = min(tolerance, 100)
        self.saveResult = saveResult
        self.lowerBound, self.upperBound = self._computeBounds()

    def _computeBounds(self):
        r, g, b = self.rgbColor
        bgrColor = (b, g, r)
        hsvColor = cv2.cvtColor(np.uint8([[bgrColor]]), cv2.COLOR_BGR2HSV)[0][0]

        lowerBound = np.clip(hsvColor - np.array([self.tolerance, self.tolerance, self.tolerance]), 0, 255).astype(np.uint8)
        upperBound = np.clip(hsvColor + np.array([self.tolerance, self.tolerance, self.tolerance]), 0, [179, 255, 255]).astype(np.uint8)

        return lowerBound, upperBound

    def _saveImage(self, image: np.ndarray):
        filename = time.strftime("ColorDetector_%d-%m-%Y_%H-%M-%S.jpg")
        cv2.imwrite(filename, image)

    def detect(self, image: np.ndarray) -> tuple:
        hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsvImage, self.lowerBound, self.upperBound)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return ()

        largestContour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largestContour)
        centerX = x + w // 2
        centerY = y + h // 2

        if self.saveResult:
            resultImage = image.copy()
            cv2.rectangle(resultImage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(resultImage, (centerX, centerY), 3, (0, 255, 0), -1)
            self._saveImage(resultImage)

        return (centerX, centerY)