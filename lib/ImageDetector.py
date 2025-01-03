import numpy as np
import cv2
import time

class ImageDetector:
    def __init__(self, targetImage: np.ndarray, threshold: float = 0.9, saveResult: bool = False):
        self.targetImage = targetImage
        self.threshold = threshold
        self.saveResult = saveResult
        self.targetHeight, self.targetWidth = self.targetImage.shape[:2]

    def _saveImage(self, image: np.ndarray):
        filename = time.strftime("ImageDetector_%d-%m-%Y_%H-%M-%S.jpg")
        cv2.imwrite(filename, image)
    
    def _markImage(self, image: np.ndarray, topLeft: tuple, bottomRight: tuple):
        image = image.copy()
        cv2.rectangle(image, topLeft, bottomRight, (0, 255, 0), 2)
        cv2.circle(image, ((topLeft[0] + bottomRight[0]) // 2, (topLeft[1] + bottomRight[1]) // 2), 3, (0, 255, 0), -1)
        return image

    def detect(self, image: np.ndarray):
        result = cv2.matchTemplate(image, self.targetImage, cv2.TM_CCOEFF_NORMED)

        locations = np.where(result >= self.threshold)
        
        if locations[0].size == 0:
            return []

        topLeft = (locations[1][0], locations[0][0])
        bottomRight = (topLeft[0] + self.targetWidth, topLeft[1] + self.targetHeight)

        if self.saveResult:
            resultImage = self._markImage(image, topLeft, bottomRight)
            self._saveImage(resultImage)

        return [topLeft, bottomRight]