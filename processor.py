import cv2
import numpy as np
from tracking_params import TrackingParameters


class VideoProcessor:
    def __init__(self, config):
        self.config = config
        self.tracking_params = TrackingParameters()
        self.position_matrix = np.empty((0, 3), dtype=int)  # frame, x, y
        self.mask = None
        self.old_gray = None
        self.frame_count = 0

    def process_video(self, video_loader):
        """Основной метод обработки видео"""
        first_frame = True

        for frame, gray_frame in video_loader.read_frames():
            self.frame_count += 1

            if first_frame:
                self._initialize_first_frame(gray_frame)
                first_frame = False
                continue

            # Обработка кадра
            self._process_frame(frame, gray_frame)

        video_loader.release()
        return self.position_matrix

    def _initialize_first_frame(self, gray_frame):
        """Инициализация для первого кадра"""
        self.old_gray = gray_frame.copy()
        self.mask = np.zeros_like(cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR))

    def _process_frame(self, frame, gray_frame):
        """Обработка одного кадра"""
        # Фоновое вычитание
        fgmask = self.tracking_params.bg_subtractor.apply(frame)
        fgmask = self.tracking_params.apply_morphology(fgmask)

        # Поиск точек для трекинга
        p0 = cv2.goodFeaturesToTrack(
            self.old_gray,
            mask=fgmask,
            **self.tracking_params.feature_params
        )

        if p0 is not None:
            self._track_points(p0, gray_frame, frame)

        # Обновление предыдущего кадра
        self.old_gray = gray_frame.copy()

    def _track_points(self, p0, gray_frame, frame):
        """Трекинг точек между кадрами"""
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            self.old_gray, gray_frame, p0, None,
            **self.tracking_params.lk_params
        )

        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Фильтрация и обновление точек
            self._update_tracking_points(good_new, good_old, frame)

    def _update_tracking_points(self, good_new, good_old, frame):
        """Обновление позиций трекируемых точек"""
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            distance = np.sqrt((a - c) ** 2 + (b - d) ** 2)

            # Фильтрация по минимальному движению
            if distance > 2.0:
                self.position_matrix = np.vstack((
                    self.position_matrix,
                    [self.frame_count, int(a), int(b)]
                ))

                # Визуализация
                color = self.tracking_params.colors[i].tolist()
                self.mask = cv2.line(
                    self.mask,
                    (int(a), int(b)),
                    (int(c), int(d)),
                    color, 2
                )
                cv2.circle(frame, (int(a), int(b)), 5, color, -1)