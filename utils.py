import time
import cv2

def print_progress(frame_count, total_frames, start_time):
    """Вывод прогресса обработки"""
    if frame_count % 30 == 0:
        elapsed = time.time() - start_time
        fps_processed = frame_count / elapsed if elapsed > 0 else 0
        progress_percent = (frame_count / total_frames) * 100
        print(f"Обработано: {frame_count}/{total_frames} ({progress_percent:.1f}%)")

def handle_keyboard_input():
    """Обработка клавиатурного ввода"""
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return 'quit'
    elif key == ord('p'):
        return 'pause'
    elif key == ord('h'):
        return 'hide'
    return None