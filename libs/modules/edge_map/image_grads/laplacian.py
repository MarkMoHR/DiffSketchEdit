import cv2


class LaplacianDetector:

    def __call__(self, img):
        return cv2.Laplacian(img, cv2.CV_64F)
