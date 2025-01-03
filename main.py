import numpy as np
import cv2
from PIL import ImageGrab
from lib import ColorDetector, ImageDetector

def getScreenshot() -> np.ndarray:
    screenshot = np.array(ImageGrab.grab(), dtype=np.uint8)
    return cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

def readImage(imagePath: str):
    return cv2.imread(filename=imagePath)

def main():
    # Test Images
    colors = readImage("test/images/colors.jpg")
    fruits = readImage("test/images/fruits.jpg")

    # Targets
    apple = readImage("test/apple.jpg")
    rgbColor = (42, 181, 148)

    colorDetector = ColorDetector(rgbColor=rgbColor, tolerance=5, saveResult=True)
    imageDetector = ImageDetector(targetImage=apple, threshold=0.9, saveResult=True)

    colorDetector.detect(colors)
    imageDetector.detect(fruits)
    
if __name__ == "__main__":
    main()