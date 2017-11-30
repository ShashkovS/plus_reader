import logging
import sys
import traceback
import numpy as np
from PyQt5.QtGui import QPixmap, QPainter, QMouseEvent
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QMenu, QSlider, QLabel
from PyQt5.QtCore import Qt

sys._excepthook = sys.excepthook


def excepthook(excType, excValue, tracebackobj):
    traceback.print_tb(tracebackobj, excType, excValue)


sys.excepthook = excepthook

VIRTUAL_BORDER_WIDTH = 5


class Label(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent=parent)
        self.page = self.parentWidget()
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
        positionx = QContextMenuEvent.x()
        positiony = QContextMenuEvent.y()
        im_pos_x, im_pos_y = list(
            map(int, self.page.image.window_coords_to_image_coords(positionx, positiony, self.width(), self.height())))
        logging.info(str(positionx) + ' ' + str(positiony) + ' -> ' + str(im_pos_x) + ' ' + str(im_pos_y))
        min_vline_dist = min(abs(im_pos_x - vl) for vl in self.page.image.coords_of_vert_lns) if self.page.image.coords_of_vert_lns\
            else float('inf')
        min_hline_dist = min(abs(im_pos_y - vl) for vl in self.page.image.coords_of_horiz_lns) if self.page.image.coords_of_horiz_lns\
            else float('inf')
        self._actions = []
        self._actions_objects = []
        if min_hline_dist <= VIRTUAL_BORDER_WIDTH * 3:
            DelHorAction = cmenu.addAction('Delete Horizontal line here')
            self._actions.append('DelHorAction')
            self._actions_objects.append(DelHorAction)
        else:
            AddHorAction = cmenu.addAction('Add Horizontal line here')
            self._actions.append('AddHorAction')
            self._actions_objects.append(AddHorAction)
        if min_vline_dist <= VIRTUAL_BORDER_WIDTH * 3:
            DelVertAction = cmenu.addAction('Delete Vertical line here')
            self._actions.append('DelVertAction')
            self._actions_objects.append(DelVertAction)
        else:
            AddVertAction = cmenu.addAction('Add Vertical line here')
            self._actions.append('AddVertAction')
            self._actions_objects.append(AddVertAction)
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
        self.page.image.coords_of_horiz_lns.append(coords[1])  # TODO: Сделать бисектом
        self.page.image.coords_of_horiz_lns.sort()
        self.page.image.find_filled_cells()
        self.page.image.initial_mark_filled_cells()
        self.page.reload_image()

    def DelHorAction(self, coords):
        logging.info('УДАЛИТЬ ГОРИЗОНТАЛЬ')
        min_dist = float('inf')
        min_line = float('inf')
        for i in self.page.image.coords_of_horiz_lns:
            dist = abs(i - coords[1])
            if dist < min_dist:
                min_dist = dist
                min_line = i
        self.page.image.coords_of_horiz_lns.remove(min_line)
        self.page.image.find_filled_cells()
        self.page.image.initial_mark_filled_cells()
        self.page.reload_image()

    def DelVertAction(self, coords):
        logging.info('УДАЛИТЬ ВЕРТИКАЛЬ')
        min_dist = float('inf')
        min_line = float('inf')
        for i in self.page.image.coords_of_vert_lns:
            dist = abs(i - coords[0])
            if dist < min_dist:
                min_dist = dist
                min_line = i
        self.page.image.coords_of_vert_lns.remove(min_line)
        self.page.image.find_filled_cells()
        self.page.image.initial_mark_filled_cells()
        self.page.reload_image()

    def AddVertAction(self, coords):
        logging.info('ДОБАВИТЬ ВЕРТИКАЛЬ')
        self.page.image.coords_of_vert_lns.append(coords[0])
        self.page.image.coords_of_vert_lns.sort()
        self.page.image.find_filled_cells()
        self.page.image.initial_mark_filled_cells()
        self.page.reload_image()

    def mousePressEvent(self, a0: QMouseEvent):
        button_pressed = a0.button()
        cursor_pos_x = int(a0.x())
        cursor_pos_y = int(a0.y())
        logging.info(str(cursor_pos_x) + ' ' + str(cursor_pos_y))
        if button_pressed == 1:
            cell_pos = self.page.image.coord_to_cell(cursor_pos_x, cursor_pos_y, self.width(), self.height())
            if cell_pos:
                self.page.image.toggle_highlight_cell(*cell_pos)
            self.page.reload_image()


class ScannedPageWidget(QWidget):
    def __init__(self, image):
        super(ScannedPageWidget, self).__init__()
        self.image = image
        self.initUi()

    def reload_image(self, *, update=True):
        self.qp.loadFromData(self.image.to_bin())
        self.lb.setPixmap(self.qp)
        if update:
            self.lb.update()

    def initUi(self):
        self.lay = QGridLayout(self)
        self.lay.setSpacing(10)
        self.lay.setContentsMargins(0, 0, 0, 0)

        self.slide = QSlider(Qt.Horizontal, self)
        self.slide.setFocusPolicy(Qt.NoFocus)
        self.slide.setTickInterval(5)
        self.slide.setMaximum(255)
        self.slide.setMinimum(0)
        self.slide.setTickPosition(QSlider.TicksBelow)
        self.slide.setTickInterval(5)
        self.slide.setValue(self.image.black_threshold)
        self.slide.valueChanged.connect(self.sliderchange)
        self.slide.sliderReleased.connect(self.valuechange)

        self.lb = Label(self)
        self.qp = QPixmap()
        self.reload_image(update=False)

        self.slval = QLabel(str(self.slide.sliderPosition()))

        self.lay.addWidget(QLabel('Change B/W Threshold'), 0, 0)
        self.lay.addWidget(self.slide, 0, 2)
        self.lay.addWidget(self.slval, 0, 9)
        self.lay.addWidget(self.lb, 1, 0, 10, 10)

        self.setLayout(self.lay)

    def sliderchange(self):
        self.slval.setText(str(self.slide.sliderPosition()))

    def valuechange(self):
        self.image.black_threshold = self.slide.sliderPosition()
        self.image.bitmap_lines_filled_cells_and_marking()
        self.reload_image()


def show(image):
    app = QApplication(sys.argv)
    _, _, screen_w, screen_h = app.primaryScreen().availableGeometry().getRect()
    img_scale = max(image.W / screen_w, image.H / screen_h)
    w_height, w_width = int(image.H / img_scale), int(image.W / img_scale),
    w = ScannedPageWidget(image)
    w.resize(w_width, w_height)
    w.show()
    app.exec_()


def feature_qt(image_cls):
    show(image_cls)
    return image_cls.filled_cells


if __name__ == '__main__':
    pass
