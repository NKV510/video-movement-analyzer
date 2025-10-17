class AnalysisConfig:
    def __init__(self):
        # Параметры обработки видео
        self.scale_factor = 0.5
        self.output_dir = 'movement_analysis_results'

        # Параметры отображения
        self.show_plots = True
        self.save_plots = True

        # Параметры обработки
        self.min_movement_threshold = 2.0