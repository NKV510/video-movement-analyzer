import cv2
import numpy as np


class TrackingParameters:
    def __init__(self):
        # Параметры для детектирования особенностей
        self.feature_params = dict(
            maxCorners=50,
            qualityLevel=0.5,
            minDistance=15,
            blockSize=9
        )

        # Параметры для оптического потока Лукаса-Канаде
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01)
        )

        # Параметры фонового вычитания
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=True
        )

        # Цвета для визуализации
        self.colors = np.random.randint(0, 255, (1000, 3))

    def apply_morphology(self, mask):
        """Применяет морфологические операции к маске"""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return mask