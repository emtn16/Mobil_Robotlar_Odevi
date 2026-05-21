import cv2
import numpy as np
import matplotlib.pyplot as plt


class Environment:
    def __init__(self, image_path):
        self.image_path = image_path
        self.grid = None
        self.start_pos = None
        self.goal_pos = None

    def load_map(self):
        img = cv2.imread(self.image_path)
        if img is None:
            raise FileNotFoundError("Harita görseli bulunamadı. Lütfen dosya yolunu kontrol et.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        self.grid = np.zeros((height, width), dtype=np.int8)

        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        lower_red = np.array([150, 0, 0])
        upper_red = np.array([255, 100, 100])
        lower_green = np.array([100, 150, 0])
        upper_green = np.array([200, 255, 100])

        mask_black = cv2.inRange(img, lower_black, upper_black)
        mask_red = cv2.inRange(img, lower_red, upper_red)
        mask_green = cv2.inRange(img, lower_green, upper_green)

        self.grid[mask_black > 0] = 1

        y_green, x_green = np.where(mask_green > 0)
        if len(x_green) > 0:
            self.start_pos = (int(np.mean(x_green)), int(np.mean(y_green)))

        y_red, x_red = np.where(mask_red > 0)
        if len(x_red) > 0:
            self.goal_pos = (int(np.mean(x_red)), int(np.mean(y_red)))