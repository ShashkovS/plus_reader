import logging
import sys
import traceback

import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QMouseEvent
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QMenu

import ImageProcessor

sys._excepthook = sys.excepthook

def excepthook(excType, excValue, tracebackobj):
    traceback.print_tb(tracebackobj)
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
        logging.info(str(positionx) + ' ' + str(positiony))
        im_pos_x, im_pos_y = page.image.window_coords_to_image_coords(positionx, positiony, self.width(), self.height())
        logging.info(str(positionx) + ' ' + str(positiony) + ' -> ' + str(im_pos_x) + ' ' + str(im_pos_y))
        min_vline_dist = min(abs(im_pos_x - vl) for vl in page.image.coords_of_vert_lns)
        min_hline_dist = min(abs(im_pos_y - vl) for vl in page.image.coords_of_horiz_lns)
        actions = []
        if min_hline_dist <= BORDER_WIDTH * 3:
            DelHorAction = cmenu.addAction('Delete Horizontal line here')
            actions.append('DelHorAction')
        else:
            AddHorAction = cmenu.addAction('Add Horizontal line here')
            actions.append('AddHorAction')
        if min_vline_dist <= BORDER_WIDTH * 3:
            DelVertAction = cmenu.addAction('Delete Vertical line here')
            actions.append('DelVertAction')
        else:
            AddVertAction = cmenu.addAction('Add Vertical line here')
            actions.append('AddVertAction')
        action = cmenu.exec_(self.mapToGlobal(QContextMenuEvent.pos()))
        print(self.pa(action))
        # TODO Поправить код на точное орпределение действия action
        # TODO Сейчас обрабатываются все сразу которые есть в меню
        # method_name = str(actions[0])
        # method = getattr(self, method_name)
        # method((im_pos_x, im_pos_y))
        # method_name = str(actions[1])
        # method = getattr(self, method_name)
        # method((im_pos_x, im_pos_y))
    # TODO Дебажный кусок кода( удалению не подлежит :-))
    # def pa(self, obj):
    #     print('*' * 100)
    #     print(type(obj), obj)
    #     dr = [x for x in dir(obj) if not x.startswith('__')]
    #     cur = {}
    #     for atr in dr:
    #         if atr == 'label':
    #             pass
    #         txt = str(obj.__getattribute__(atr)).replace('\n', '')
    #         if txt in ['None']:
    #             continue
    #         if txt.startswith('<') and txt.endswith('>') and ' at ' in txt:
    #             continue
    #         cur[atr] = txt
    #     return cur

    def AddHorAction(self, coords):
        logging.info('ДОБАВИТЬ ГОРИЗОНТАЛЬ')
        pass

    def DelHorAction(self, coords):
        logging.info('УДАЛИТЬ ГОРИЗОНТАЛЬ')
        pass

    def DelVertAction(self, coords):
        logging.info('УДАЛИТЬ ВЕРТИКАЛЬ')
        pass

    def AddVertAction(self, coords):
        logging.info('ДОБАВИТЬ ВЕРТИКАЛЬ')
        pass


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


def feature_qt(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns):
    global image
    image = ImageProcessor.ImageProcessor(gray_np_image, filled_cells, coords_of_horiz_lns, coords_of_vert_lns)
    show(image)
    return filled_cells


if __name__ == '__main__':
    pass