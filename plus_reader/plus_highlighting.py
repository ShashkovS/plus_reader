import logging
import sys
import traceback
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QMouseEvent
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMenu

import ImageProcessor
from cell_recognizer import find_filled_cells

sys._excepthook = sys.excepthook

def excepthook(excType, excValue, tracebackobj):
    traceback.print_tb(tracebackobj, excType, excValue)
sys.excepthook = excepthook


FILL_COLOR = np.array([[[0, 255, 255]]], dtype=np.uint8)
BORDER_COLOR = np.array([[[0, 0, 255]]], dtype=np.uint8)
BORDER_WIDTH = 5
MAX_SIZE = 800




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

    def contextMenuEvent(self, QContextMenuEvent):
        cmenu = QMenu(self)
        page = self.parentWidget()
        positionx = QContextMenuEvent.x()
        positiony = QContextMenuEvent.y()
        im_pos_x, im_pos_y = list(map(int, page.image.window_coords_to_image_coords(positionx, positiony, self.width(), self.height())))
        logging.info(str(positionx) + ' ' + str(positiony) + ' -> ' + str(im_pos_x) + ' ' + str(im_pos_y))
        min_vline_dist = min(abs(im_pos_x - vl) for vl in page.image.coords_of_vert_lns)
        min_hline_dist = min(abs(im_pos_y - vl) for vl in page.image.coords_of_horiz_lns)
        self._actions = []
        self._actions_objects = []
        if min_hline_dist <= BORDER_WIDTH:
            DelHorAction = cmenu.addAction('Delete Horizontal line here')
            self._actions.append('DelHorAction')
            self._actions_objects.append(DelHorAction )
        else:
            AddHorAction = cmenu.addAction('Add Horizontal line here')
            self._actions.append('AddHorAction')
            self._actions_objects.append(AddHorAction )
        if min_vline_dist <= BORDER_WIDTH:
            DelVertAction = cmenu.addAction('Delete Vertical line here')
            self._actions.append('DelVertAction')
            self._actions_objects.append(DelVertAction )
        else:
            AddVertAction = cmenu.addAction('Add Vertical line here')
            self._actions.append('AddVertAction')
            self._actions_objects.append(AddVertAction )
        action = cmenu.exec_(self.mapToGlobal(QContextMenuEvent.pos()))
        if action:
            selected_action_index = self._actions_objects.index(action)
            selected_action = self._actions[selected_action_index]
            logging.info(str(selected_action))
            # TODO работающих методов ещё нет поэтому этот кусок пока не нужен
            method = getattr(self, selected_action)
            method((im_pos_x, im_pos_y))

    def AddHorAction(self, coords):
        logging.info('ДОБАВИТЬ ГОРИЗОНТАЛЬ')
        page = self.parentWidget()
        page.image.coords_of_horiz_lns.append(coords[1])  # TODO: Сделать бисектом
        page.image.coords_of_horiz_lns.sort()
        page.image.filled_cells = find_filled_cells(page.image.image_without_lines,
                                                    page.image.coords_of_horiz_lns, page.image.coords_of_vert_lns)
        page.image.bgr_img_with_highlights = cv2.cvtColor(page.image.gray_np_image, cv2.COLOR_GRAY2BGR)
        page.image.initial_mark_filled_cells()
        # TODO Перераспознование
        page.qp.loadFromData(page.image.to_bin())
        page.lb.setPixmap(page.qp)
        page.lb.update()


    def DelHorAction(self, coords):
        logging.info('УДАЛИТЬ ГОРИЗОНТАЛЬ')
        page = self.parentWidget()
        page.image.coords_of_horiz_lns.remove(coords[1])
        page.image.filled_cells = find_filled_cells(page.image.image_without_lines,
                                                    page.image.coords_of_horiz_lns, page.image.coords_of_vert_lns)
        page.image.bgr_img_with_highlights = cv2.cvtColor(page.image.gray_np_image, cv2.COLOR_GRAY2BGR)
        page.image.initial_mark_filled_cells()
        page.qp.loadFromData(page.image.to_bin())
        page.lb.setPixmap(page.qp)
        page.lb.update()

    def DelVertAction(self, coords):
        logging.info('УДАЛИТЬ ВЕРТИКАЛЬ')
        page = self.parentWidget()
        # TODO Тут надо точное определение линии по приблизительным координатам
        page.image.coords_of_vert_lns.remove(coords[0])
        page.image.filled_cells = find_filled_cells(page.image.image_without_lines,
                                                    page.image.coords_of_horiz_lns, page.image.coords_of_vert_lns)
        page.image.bgr_img_with_highlights = cv2.cvtColor(page.image.gray_np_image, cv2.COLOR_GRAY2BGR)
        page.image.initial_mark_filled_cells()
        page.qp.loadFromData(page.image.to_bin())
        page.lb.setPixmap(page.qp)
        page.lb.update()

    def AddVertAction(self, coords):
        logging.info('ДОБАВИТЬ ВЕРТИКАЛЬ')
        page = self.parentWidget()
        page.image.coords_of_vert_lns.append(coords[0])
        page.image.coords_of_vert_lns.sort()
        page.image.filled_cells = find_filled_cells(page.image.image_without_lines,
                                                    page.image.coords_of_horiz_lns, page.image.coords_of_vert_lns)
        page.image.bgr_img_with_highlights = cv2.cvtColor(page.image.gray_np_image, cv2.COLOR_GRAY2BGR)
        page.image.initial_mark_filled_cells()

        # TODO Перераспознование
        page.qp.loadFromData(page.image.to_bin())
        page.lb.setPixmap(page.qp)
        page.lb.update()


    def mousePressEvent(self, a0: QMouseEvent):
        button_pressed = a0.button()
        page = self.parentWidget()
        cursor_pos_x = int(a0.x())
        cursor_pos_y = int(a0.y())
        logging.info(str(cursor_pos_x) + ' ' + str(cursor_pos_y))
        if button_pressed == 1:
            cell_pos = page.image.coord_to_cell(cursor_pos_x, cursor_pos_y, self.width(), self.height())
            if cell_pos:
                page.image.toggle_highlight_cell(*cell_pos)
            page.qp.loadFromData(page.image.to_bin())
            page.lb.setPixmap(page.qp)
            page.lb.update()



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

def show(image):
    mx = max(image.H, image.W)
    w_height, w_width = int(image.H/mx*MAX_SIZE), int(image.W/mx*MAX_SIZE),
    app = QApplication(sys.argv)
    w = ScannedPageWidget(image)
    w.resize(w_width, w_height)
    # w.setFixedSize(w_width, w_height)
    w.show()
    app.exec_()


def feature_qt(image_cls):
    show(image_cls)
    return image_cls.filled_cells


if __name__ == '__main__':
    pass