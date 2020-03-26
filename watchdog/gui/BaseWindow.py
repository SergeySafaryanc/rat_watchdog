from PyQt5 import QtWidgets


class BaseWindow(QtWidgets.QWidget):
    def __init__(self, screen=0, fullScreen=False, center=True, size=(None, None)):
        super().__init__()
        self.fullScreen = fullScreen
        desktop = QtWidgets.QApplication.desktop()
        RectScreen = desktop.screenGeometry(screen)
        w = size[0] if size[0] else RectScreen.width()
        h = size[1] if size[1] else RectScreen.height()
        x, y = center and ((RectScreen.width() - w) // 2, (RectScreen.height() - h) // 2) or (0, 0)
        x, y = x + RectScreen.left(), y + RectScreen.top()
        self.setGeometry(x, y, w, h)
