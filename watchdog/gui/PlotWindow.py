from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QScrollArea
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from configs.watchdog_config import *
from watchdog.gui.BaseWindow import BaseWindow
import numpy as np
import threading


class PlotWindow(BaseWindow):
    def __init__(self, screen=0, fullScreen=False, center=True, size=(None, None), channel_pairs=channel_pairs_to_corr):
        super().__init__(screen, fullScreen, center, size)
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint)
        self.channel_pairs = channel_pairs
        self.figure, self.axis = plt.subplots(2, 1, figsize=(15.5, 70))
        self.figure.tight_layout(h_pad=0.001)
        for ax in self.axis:
            ax.set_axis_off()
            ax.tick_params(labeltop=False, labelbottom=False, labelright=False, labelleft=False, bottom=False,
                           top=False, left=False, right=False)
        plt.subplots_adjust(wspace=0, hspace=0)

        self.canvas = FigureCanvas(self.figure)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.canvas)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.scroll)

        self.setLayout(layout)
        self.fullScreen and self.showMaximized() or self.show()
        self.dataCorr = []
        self.dataBR = []
        self.vlines = []
        self.hlines = []
        self.initPlot()

    def initPlot(self):
        for i, ax in enumerate(self.axis):
            ax.tick_params(bottom=False, top=False, left=False, right=False)
            if i == 0:
                ax.plot([], color='red', label="breath")
                ax.legend(loc="upper left")
            elif i == 1:
                ax.plot([], color='green', label="corr")
                ax.legend(loc="upper left")

    def plot(self):
        proc = threading.Thread(target=self.renderPlotInThread)
        proc.daemon = False
        proc.start()

    def renderPlotInThread(self):
        for i, ax in enumerate(self.axis):
            ax.clear()
            if i == 0:
                ax.plot(self.dataBR, color='red', label="breath")
                ax.legend(loc="upper left")
            elif i == 1:
                ax.plot(np.mean(self.dataCorr, axis=1), color='green', label="corr")
                ax.legend(loc="upper left")
            data_len = len(self.dataBR)
            ax.set_axis_off()
            x_min, x_max = (data_len - 60, data_len) if (data_len > 60) else (0, data_len)
            x_min = 0
            ax.set_xlim(x_min, x_max)
            ax.vlines(self.vlines, 0, 1)
            ax.hlines(self.hlines, x_min, x_max)
        self.canvas.draw()

    def plotSig(self):
        ax = self.axis[1]
        ax.clear()
        ax.plot(self.sig, color='red', label="stimuls")
        data_len = self.sig.shape[0]
        ax.set_ylim(0, 18)
        lenView = 200000
        x_min, x_max = (data_len - lenView, data_len) if (data_len > lenView) else (0, data_len)
        ax.set_xlim(x_min, x_max)
        ax.vlines(self.vlines, 0, 1)
        ax.hlines(self.hlines, x_min, x_max)
        self.canvas.draw()

    def addPoint(self, corr, br):
        self.dataCorr = corr
        self.dataBR = br
        self.plot()

    def addPointSig(self, sig):
        self.sig = sig
        self.plotSig()

    def __addHVLines(self, index, lines, value):
        if index is None:
            lines.append(value)
            return len(lines) - 1
        lines[index] = value
        return index

    def addVLine(self, value=None, index=None):
        value = value if value else len(self.data)
        return self.__addHVLines(index, self.vlines, value)

    def addHLine(self, value=None, index=None):
        value = value if value else self.data[-1]
        return self.__addHVLines(index, self.hlines, value)
