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
BORDER_WIDTH = 7


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


def show(bin_image, im_height, im_width):
    mx = max(im_height, im_width)
    w_height, w_width = int(im_height/mx*640), int(im_width/mx*640),
    print(w_width, w_height)

    app = QApplication(sys.argv)
    w = ScannedPageWidget(bin_image)
    w.resize(w_height, w_width)
    w.show()
    sys.exit(app.exec_())


def toggle_highlight_cell(i, j, image, filled_cells, horizontal_coords, vertical_coords, *, initial_mode=False):
    FILL_COLOR = np.array([0, 255, 255], dtype=np.uint8)
    ALPHA = .3
    bw = BORDER_WIDTH // 2
    h1, h2, v1, v2 = horizontal_coords[i]+bw, horizontal_coords[i + 1]-bw, vertical_coords[j]+bw, vertical_coords[j + 1]-bw
    if filled_cells[i][j]:
        image[h1:h2, v1:v2, :] = (1-ALPHA) * image[h1:h2, v1:v2, :] + ALPHA*FILL_COLOR  # GBR, so it's yellow
    elif not initial_mode:
        image[h1:h2, v1:v2, :] = (image[h1:h2, v1:v2, :] - ALPHA*FILL_COLOR) / (1-ALPHA)
    if not initial_mode:
        filled_cells[i][j] ^= filled_cells[i][j]


def mark_filled_cells(gray_np_image, filled_cells, horizontal_coords, vertical_coords, show_borders=True):
    """Отмаркировать ячейки, которые распознались как заполненные
    (заливка жёлтым с прозрачностью 0.3)"""
    colored = cv2.cvtColor(gray_np_image, cv2.COLOR_GRAY2BGR)
    # Каждую заполненную ячейку подкрашиваем жёлтым
    for i in range(len(horizontal_coords) - 1):
        for j in range(len(vertical_coords) - 1):
            toggle_highlight_cell(i, j, colored, filled_cells, horizontal_coords, vertical_coords, initial_mode=True)
    # Добавляем распознанные границы красным:
    if show_borders:
        bw = BORDER_WIDTH // 2
        for v_b in vertical_coords:
            v1, v2 = v_b - bw, v_b + bw
            colored[:, v1:v2, :] = .7*colored[:, v1:v2, :] + .3*BORDER_COLOR
        for h_b in horizontal_coords:
            h1, h2 = h_b - bw, h_b + bw
            colored[h1:h2, :, :] = .7*colored[h1:h2, :, :] + .3*BORDER_COLOR
    return colored



def feature_qt(gray_np_image, filled_cells, horizontal_coords, vertical_coords):
    highlighted_image = mark_filled_cells(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    retval, bin_image = cv2.imencode('.png', highlighted_image)
    im_height, im_width = gray_np_image.shape
    show(bin_image, im_height, im_width)
    return filled_cells


if __name__ == '__main__':
    pass