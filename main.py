import cv2
import os
from video_loader import VideoLoader
from processor import VideoProcessor
from data_analyzer import MovementAnalyzer
from plot_generator import PlotGenerator
from config import AnalysisConfig


def main():
    # Конфигурация
    config = AnalysisConfig()
    video_path = "" #Необходимо указать путь к видеофайлу в ковычках

    if not os.path.exists(video_path):
        print(f"Ошибка: файл '{video_path}' не найден!")
        return

    # Загрузка видео
    print("Загрузка видео...")
    video_loader = VideoLoader(video_path, config.scale_factor)
    video_info = video_loader.get_video_info()
    print(f"Информация о видео: {video_info}")

    # Обработка видео
    print("Обработка видео...")
    processor = VideoProcessor(config)
    tracking_data = processor.process_video(video_loader)

    if len(tracking_data) == 0:
        print("Не обнаружено значимого движения!")
        return

    # Анализ данных
    print("Анализ данных...")
    analyzer = MovementAnalyzer()
    analysis_results = analyzer.analyze(tracking_data)

    # Визуализация
    print("Создание графиков...")
    plotter = PlotGenerator(config.output_dir)
    plotter.create_all_plots(analysis_results['processed_data'])

    # Сохранение результатов
    analyzer.save_results(analysis_results, config.output_dir)

    # Вывод статистики
    analyzer.print_statistics(analysis_results['statistics'])

    print(f"\nАнализ завершен! Результаты сохранены в '{config.output_dir}'")


if __name__ == "__main__":
    main()