import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, grid_map, start_pos, goal_pos):
        self.grid_map = grid_map
        self.start = start_pos
        self.goal = goal_pos

    def _draw_map(self, ax, title, path, metrics):
        ax.imshow(self.grid_map, cmap='binary', origin='upper')
        ax.plot(self.start[0], self.start[1], 'go', markersize=8, label='Başlangıç')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=8, label='Hedef')

        if path:
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]
            ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Planlanan Yol')
            text_str = f"Süre: {metrics['time']:.3f} sn\nYol: {metrics['length']:.1f} px"
            ax.text(0.05, 0.95, text_str, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "YOL BULUNAMADI", transform=ax.transAxes, color='red',
                    fontsize=12, ha='center', fontweight='bold')

        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.axis('off')

    def show_single(self, title, path, metrics):
        fig, ax = plt.subplots(figsize=(8, 6))
        self._draw_map(ax, title, path, metrics)
        plt.tight_layout()
        plt.show()

    def show_comparison(self, title1, path1, metrics1, title2, path2, metrics2):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Algoritma ve Kinematik Karşılaştırması", fontsize=16)
        self._draw_map(ax1, title1, path1, metrics1)
        self._draw_map(ax2, title2, path2, metrics2)
        plt.tight_layout()
        plt.show()