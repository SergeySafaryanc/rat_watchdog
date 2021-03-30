from PyQt5 import QtWidgets
import sys

from watchdog.gui.MainWindow import MainWindow
from watchdog.gui.PlotWindow import PlotWindow


class GUI(object):
    def __init__(self, channel_pairs):
        self.app = QtWidgets.QApplication([])
        self.mainWindow = MainWindow(screen=0, fullScreen=True, channel_pairs=channel_pairs)
        self.mainWindow.closeEvent = self.quit
        self.mainWindow.start_button.clicked.connect(self.validation)

        # self.plotWindow = PlotWindow(screen=1, center=True, fullScreen=True, channel_pairs=channel_pairs)

    def start(self):
        r = self.app.exec()
        sys.exit()

    def quit(self, e):
        self.app.exit()

    def validation(self):
        # c = Config()
        print("Hello!")
