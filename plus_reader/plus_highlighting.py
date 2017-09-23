import logging
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter
import cv2  # pip install --upgrade opencv-python
from PIL import Image  # pip install --upgrade pillow
from io import BytesIO

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


def mark_filled_cells(gray_np_image, filled_cells, horizontal_coords, vertical_coords, show_borders=True):
    """Отмаркировать ячейки, которые распознались как заполненные
    (заливка жёлтым с прозрачностью 0.3)"""
    clrd = cv2.cvtColor(gray_np_image, cv2.COLOR_GRAY2BGR)
    clrd_r = clrd.copy()
    # Каждую заполненную ячейку подкрашиваем жёлтым
    for i in range(len(horizontal_coords) - 1):
        for j in range(len(vertical_coords) - 1):
            if filled_cells[i][j]:
                clrd_r[horizontal_coords[i]:horizontal_coords[i + 1], vertical_coords[j]:vertical_coords[j + 1], :] = [0, 255, 255]  #
    # Добавляем распознанные границы красным:
    if show_borders:
        L_W = 3
        for h_b in horizontal_coords:
            clrd_r[:, h_b - L_W:h_b + L_W:, :] = [255, 0, 0]
        for v_b in vertical_coords:
            clrd_r[v_b - L_W:v_b + L_W:, :, :] = [255, 0, 0]
    return cv2.addWeighted(clrd, .7, clrd_r, .3, 0)


def feature_qt(gray_np_image, filled_cells, horizontal_coords, vertical_coords):
    highlighted_image = mark_filled_cells(gray_np_image, filled_cells, horizontal_coords, vertical_coords)
    retval, bin_image = cv2.imencode('.png', highlighted_image)
    im_height, im_width = gray_np_image.shape
    show(bin_image, im_height, im_width)
    return filled_cells


if __name__ == '__main__':
    pass