import pandas as pd
import numpy as np
from scipy import signal
import os


class MovementAnalyzer:
    def __init__(self):
        pass

    def analyze(self, position_matrix):
        """Анализ данных движения"""
        if len(position_matrix) == 0:
            return {}

        # Создание DataFrame
        df = pd.DataFrame(position_matrix, columns=['frame_num', 'pos_x', 'pos_y'])

        # Обработка данных
        processed_data = self._process_movement_data(df)

        # Расчет статистики
        statistics = self._calculate_statistics(processed_data)

        return {
            'raw_data': position_matrix,
            'processed_data': processed_data,
            'statistics': statistics
        }

    def _process_movement_data(self, df):
        """Обработка данных движения"""
        # Группировка по кадрам
        df_grouped = df.groupby('frame_num').agg({
            'pos_x': 'mean',
            'pos_y': 'mean'
        }).reset_index()

        # Вычисление изменений
        df_grouped['delta_x'] = df_grouped['pos_x'].diff()
        df_grouped['delta_y'] = df_grouped['pos_y'].diff()
        df_grouped['delta_total'] = np.sqrt(
            df_grouped['delta_x'] ** 2 + df_grouped['delta_y'] ** 2
        )

        # Сглаживание
        if len(df_grouped) > 5:
            window_size = min(5, len(df_grouped) // 2)
            df_grouped['delta_total_smoothed'] = signal.medfilt(
                df_grouped['delta_total'].fillna(0),
                kernel_size=window_size
            )
        else:
            df_grouped['delta_total_smoothed'] = df_grouped['delta_total']

        # Кумулятивное расстояние
        df_grouped['cumulative_distance'] = df_grouped['delta_total_smoothed'].cumsum()

        return df_grouped

    def _calculate_statistics(self, df):
        """Расчет статистики"""
        if len(df) == 0:
            return {}

        return {
            'total_distance': df['cumulative_distance'].iloc[-1],
            'mean_velocity': df['delta_total_smoothed'].mean(),
            'max_velocity': df['delta_total_smoothed'].max(),
            'median_velocity': df['delta_total_smoothed'].median(),
            'total_frames_analyzed': len(df),
            'total_movement_points': len(df)
        }

    def save_results(self, analysis_results, output_dir):
        """Сохранение результатов"""
        os.makedirs(output_dir, exist_ok=True)

        # Сохранение сырых данных
        np.savetxt(
            os.path.join(output_dir, 'movement_data.csv'),
            analysis_results['raw_data'],
            delimiter=',',
            header='frame_num,pos_x,pos_y',
            fmt='%d'
        )

        # Сохранение обработанных данных
        analysis_results['processed_data'].to_csv(
            os.path.join(output_dir, 'smoothed_movement_data.csv'),
            index=False
        )

    def print_statistics(self, statistics):
        """Вывод статистики"""
        print("\n" + "=" * 50)
        print("СТАТИСТИКА ДВИЖЕНИЯ")
        print("=" * 50)
        for key, value in statistics.items():
            if isinstance(value, float):
                print(f"{key.replace('_', ' ').title()}: {value:.2f}")
            else:
                print(f"{key.replace('_', ' ').title()}: {value}")