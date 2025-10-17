import cv2
import os


class VideoLoader:
    def __init__(self, video_path, scale_factor=0.5):
        self.video_path = video_path
        self.scale_factor = scale_factor
        self.cap = None
        self._open_video()

    def _open_video(self):
        """Открывает видео файл"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Видео файл не найден: {self.video_path}")

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Не удалось открыть видео файл")

    def get_video_info(self):
        """Возвращает информацию о видео"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration_sec': total_frames / fps if fps > 0 else 0
        }

    def read_frames(self):
        """Генератор для чтения кадров"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Масштабирование
            scaled_frame = cv2.resize(
                frame,
                (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor
            )

            # Конвертация в grayscale
            gray_frame = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

            yield scaled_frame, gray_frame

    def release(self):
        """Освобождение ресурсов"""
        if self.cap:
            self.cap.release()