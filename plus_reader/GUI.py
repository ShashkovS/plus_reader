import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QPainter

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


class Widget(QWidget):
    def __init__(self, imgpath, parent=None):
        QWidget.__init__(self, parent=parent)
        lay = QVBoxLayout(self)
        lb = Label(self)
        lb.setPixmap(QPixmap(imgpath))
        lay.addWidget(lb)

def show(imgpath):
    app = QApplication(sys.argv)
    w = Widget(imgpath)
    w.resize(640, 480)
    w.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    pass