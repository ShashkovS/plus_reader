import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QMouseEvent, QCursor
from PyQt5.QtCore import QMargins
from bisect import bisect_left
import cv2  # pip install --upgrade opencv-python
import numpy as np
import logging
import sys
import traceback
sys._excepthook = sys.excepthook

def excepthook(excType, excValue, tracebackobj):
    traceback.print_tb(tracebackobj)
sys.excepthook = excepthook


FILL_COLOR = np.array([[[0, 255, 255]]], dtype=np.uint8)
BORDER_COLOR = np.array([[[0, 0, 255]]], dtype=np.uint8)
BORDER_WIDTH = 5
MAX_SIZE = 800


class ImageProcessor():
    def __init__(self, image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns, *, show_borders=True):
        """Сохраняем все данные в атрибутах, производим первичную раскраску"""
        self.image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.H, self.W, *_ = image.shape
        self.filled_cells = filled_cells
        self.coords_of_horiz_lns = coords_of_horiz_lns
        self.coords_of_vert_lns = coords_of_vert_lns
        self.BW = BORDER_WIDTH if show_borders else 0
        self.initial_mark_filled_cells()

    def toggle_highlight_cell(self, x_vert_ind, y_horiz_ind, *, initial_mode=False):
        """Маркировка и снятие маркировки ячейки"""
        FILL_COLOR = np.array([0, 255, 255], dtype=np.uint8)
        ALPHA = .3
        bw = self.BW
        if x_vert_ind < 0 or x_vert_ind >= len(self.coords_of_vert_lns) - 1:
            logging.error('Cell x-index out of range: {}, cor.range: {}-{}'.format(x_vert_ind, 0, len(self.coords_of_vert_lns) - 1))
            return
        if y_horiz_ind < 0 or y_horiz_ind >= len(self.coords_of_horiz_lns) - 1:
            logging.error('Cell y-index out of range: {}, cor.range: {}-{}'.format(y_horiz_ind, 0, len(self.coords_of_horiz_lns) - 1))
            return
        h1, h2, v1, v2 = self.coords_of_horiz_lns[y_horiz_ind] + bw, self.coords_of_horiz_lns[y_horiz_ind + 1] - bw, \
                         self.coords_of_vert_lns[x_vert_ind] + bw, self.coords_of_vert_lns[x_vert_ind + 1] - bw
        if not initial_mode ^ self.filled_cells[y_horiz_ind][x_vert_ind]:  # initial_mode инвертирует логику
            self.image[h1:h2, v1:v2, :] = (1-ALPHA) * self.image[h1:h2, v1:v2, :] + ALPHA*FILL_COLOR  # GBR, so it's yellow
        elif not initial_mode and self.filled_cells[y_horiz_ind][x_vert_ind]:
            self.image[h1:h2, v1:v2, :] = (self.image[h1:h2, v1:v2, :] - ALPHA*FILL_COLOR) / (1-ALPHA)
        if not initial_mode:
            self.filled_cells[y_horiz_ind][x_vert_ind] ^= True

    def initial_mark_filled_cells(self):
        """Выполнить первичную маркировку всех заполненных ячеек"""
        for y_horiz_ind in range(len(self.coords_of_horiz_lns) - 1):
            for x_vert_ind in range(len(self.coords_of_vert_lns) - 1):
                self.toggle_highlight_cell(x_vert_ind, y_horiz_ind, initial_mode=True)
        # Добавляем распознанные границы красным:
        if self.BW:
            bw = self.BW
            for v_b in self.coords_of_vert_lns:
                v1, v2 = v_b - bw, v_b + bw
                self.image[:, v1:v2, :] = .7 * self.image[:, v1:v2, :] + .3 * BORDER_COLOR
            for h_b in self.coords_of_horiz_lns:
                h1, h2 = h_b - bw, h_b + bw
                self.image[h1:h2, :, :] = .7 * self.image[h1:h2, :, :] + .3 * BORDER_COLOR

    def coord_to_cell(self, x, y, w, h):
        real_x_coord = x * self.W / w
        real_y_coord = y * self.H / h
        x_ind = bisect_left(self.coords_of_vert_lns, real_x_coord) - 1
        y_ind = bisect_left(self.coords_of_horiz_lns, real_y_coord) - 1
        if x_ind < 0 or y_ind < 0 or x_ind >= len(self.coords_of_vert_lns) - 1 or y_ind >= len(self.coords_of_horiz_lns) - 1:
            res = None
        else:
            res = [x_ind, y_ind]
        logging.info(str(res))
        return res


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
    def __init__(self, image, parent=None):
        QWidget.__init__(self, parent=parent)
        self.image = image
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(0, 0, 0, 0)
        self.lb = Label(self)
        self.qp = QPixmap()
        self.qp.loadFromData(self.image.to_bin())
        self.lb.setPixmap(self.qp)
        self.lay.addWidget(self.lb)

    def mousePressEvent(self, a0: QMouseEvent):
        cursor_pos_x = int(a0.x())
        cursor_pos_y = int(a0.y())
        logging.info(str(cursor_pos_x) + ' ' + str(cursor_pos_y))
        cell_pos = self.image.coord_to_cell(cursor_pos_x, cursor_pos_y, self.width(), self.height())
        if cell_pos:
            self.image.toggle_highlight_cell(*cell_pos)
        self.qp.loadFromData(self.image.to_bin())
        self.lb.setPixmap(self.qp)
        self.lb.update()

def show(image):
    mx = max(image.H, image.W)
    w_height, w_width = int(image.H/mx*MAX_SIZE), int(image.W/mx*MAX_SIZE),
    app = QApplication(sys.argv)
    w = ScannedPageWidget(image)
    w.resize(w_width, w_height)
    # w.setFixedSize(w_width, w_height)
    w.show()
    app.exec_()


def feature_qt(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns):
    global image
    image = ImageProcessor(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns)
    show(image)
    return filled_cells


if __name__ == '__main__':
    pass