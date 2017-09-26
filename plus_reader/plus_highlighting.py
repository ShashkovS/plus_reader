import logging
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter
import cv2  # pip install --upgrade opencv-python
import numpy as np
from PIL import Image  # pip install --upgrade pillow
from io import BytesIO

FILL_COLOR = np.array([[[0, 255, 255]]], dtype=np.uint8)
BORDER_COLOR = np.array([[[0, 0, 255]]], dtype=np.uint8)
BORDER_WIDTH = 5


class ImageProcessor():
    def __init__(self, image, filled_cells, horizontal_coords, vertical_coords, *, show_borders=True):
        """Сохраняем все данные в атрибутах, производим первичную раскраску"""
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.H, self.W, *_ = image.shape
        self.filled_cells = filled_cells
        self.horizontal_coords = horizontal_coords
        self.vertical_coords = vertical_coords
        self.BW = BORDER_WIDTH if show_borders else 0
        self.initial_mark_filled_cells()

    def toggle_highlight_cell(self, i, j, *, initial_mode=False):
        """Маркировка и снятие маркировки ячейки"""
        FILL_COLOR = np.array([0, 255, 255], dtype=np.uint8)
        ALPHA = .3
        bw = self.BW
        h1, h2, v1, v2 = self.horizontal_coords[i]+bw, self.horizontal_coords[i + 1]-bw, \
                         self.vertical_coords[j]+bw, self.vertical_coords[j + 1]-bw
        if not initial_mode ^ self.filled_cells[i][j]:  # initial_mode инвертирует логику
            self.image[h1:h2, v1:v2, :] = (1-ALPHA) * self.image[h1:h2, v1:v2, :] + ALPHA*FILL_COLOR  # GBR, so it's yellow
        elif not initial_mode and self.filled_cells[i][j]:
            self.image[h1:h2, v1:v2, :] = (self.image[h1:h2, v1:v2, :] - ALPHA*FILL_COLOR) / (1-ALPHA)
        if not initial_mode:
            self.filled_cells[i][j] ^= self.filled_cells[i][j]

    def initial_mark_filled_cells(self):
        """Выполнить первичную маркировку всех заполненных ячеек"""
        for i in range(len(self.horizontal_coords) - 1):
            for j in range(len(self.vertical_coords) - 1):
                self.toggle_highlight_cell(i, j, initial_mode=True)
        # Добавляем распознанные границы красным:
        if self.BW:
            bw = self.BW
            for v_b in self.vertical_coords:
                v1, v2 = v_b - bw, v_b + bw
                self.image[:, v1:v2, :] = .7 * self.image[:, v1:v2, :] + .3 * BORDER_COLOR
            for h_b in self.horizontal_coords:
                h1, h2 = h_b - bw, h_b + bw
                self.image[h1:h2, :, :] = .7 * self.image[h1:h2, :, :] + .3 * BORDER_COLOR

    def to_bin(self):
        """Конвертнуть текущее состояние кортинки в бинарную строку для передачи в QT"""
        retval, bin_image = cv2.imencode('.png', self.image)
        return bin_image


class Label(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.p = None

    def setPixmap(self, p):
        self.p = p

    def paintEvent(self, event):
        if self.p:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self.p)


class ScannedPageWidget(QWidget):
    def __init__(self, bin_image, parent=None):
        QWidget.__init__(self, parent=parent)
        lay = QVBoxLayout(self)
        lb = Label(self)
        qp = QPixmap()
        qp.loadFromData(bin_image)
        lb.setPixmap(qp)
        lay.addWidget(lb)


def show(image):
    MAX_SIZE = 800
    mx = max(image.H, image.W)
    w_height, w_width = int(image.H/mx*MAX_SIZE), int(image.W/mx*MAX_SIZE),

    app = QApplication(sys.argv)
    w = ScannedPageWidget(image.to_bin())
    w.resize(w_height, w_width)
    w.show()
    sys.exit(app.exec_())


def feature_qt(gray_np_image, filled_cells, horizontal_coords, vertical_coords):
    image = ImageProcessor(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    show(image)
    return filled_cells


if __name__ == '__main__':
    pass