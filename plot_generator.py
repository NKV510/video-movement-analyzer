import matplotlib.pyplot as plt
import pandas as pd
import os


class PlotGenerator:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def create_all_plots(self, df):
        """Создание всех графиков"""
        if len(df) == 0:
            return

        # Основной график с несколькими subplots
        self._create_main_plot(df)

        # Отдельные графики
        self._create_trajectory_plot(df)
        self._create_velocity_plot(df)
        self._create_cumulative_plot(df)

    def _create_main_plot(self, df):
        """Создание основного графика с 4 subplots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Траектория движения
        self._plot_trajectory(ax1, df)

        # 2. Скорость движения
        self._plot_velocity(ax2, df)

        # 3. Кумулятивное расстояние
        self._plot_cumulative_distance(ax3, df)

        # 4. Распределение скоростей
        self._plot_velocity_distribution(ax4, df)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, 'analysis_results.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()

    def _create_trajectory_plot(self, df):
        """Создание отдельного графика траектории"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_trajectory(ax, df)
        plt.savefig(
            os.path.join(self.output_dir, 'trajectory.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _create_velocity_plot(self, df):
        """Создание отдельного графика скорости"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_velocity(ax, df)
        plt.savefig(
            os.path.join(self.output_dir, 'velocity.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _create_cumulative_plot(self, df):
        """Создание отдельного графика кумулятивного расстояния"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._plot_cumulative_distance(ax, df)
        plt.savefig(
            os.path.join(self.output_dir, 'cumulative_distance.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def _plot_trajectory(self, ax, df):
        """Построение графика траектории"""
        ax.plot(df['pos_x'], df['pos_y'], 'b-', alpha=0.7, linewidth=2)
        if len(df) > 0:
            ax.scatter(df['pos_x'].iloc[0], df['pos_y'].iloc[0],
                       color='green', s=100, label='Start')
            ax.scatter(df['pos_x'].iloc[-1], df['pos_y'].iloc[-1],
                       color='red', s=100, label='End')
        ax.set_xlabel('X координата')
        ax.set_ylabel('Y координата')
        ax.set_title('Траектория движения')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_velocity(self, ax, df):
        """Построение графика скорости"""
        ax.plot(df['frame_num'], df['delta_total_smoothed'], 'r-', linewidth=2)
        ax.set_xlabel('Номер кадра')
        ax.set_ylabel('Скорость (пиксели/кадр)')
        ax.set_title('Скорость движения (сглаженная)')
        ax.grid(True, alpha=0.3)

    def _plot_cumulative_distance(self, ax, df):
        """Построение графика кумулятивного расстояния"""
        ax.plot(df['frame_num'], df['cumulative_distance'], 'purple', linewidth=2)
        ax.set_xlabel('Номер кадра')
        ax.set_ylabel('Общее расстояние (пиксели)')
        ax.set_title('Общее пройденное расстояние')
        ax.grid(True, alpha=0.3)

    def _plot_velocity_distribution(self, ax, df):
        """Построение гистограммы распределения скоростей"""
        ax.hist(df['delta_total_smoothed'].dropna(), bins=20,
                alpha=0.7, color='orange', edgecolor='black')
        ax.set_xlabel('Скорость (пиксели/кадр)')
        ax.set_ylabel('Частота')
        ax.set_title('Распределение скоростей')
        ax.grid(True, alpha=0.3)