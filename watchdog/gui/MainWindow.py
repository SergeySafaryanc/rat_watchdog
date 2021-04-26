from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QMessageBox, QFormLayout, QLineEdit, QCheckBox, QPushButton, QLabel
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QColor, QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from configs.json_parser_configuration import Config
from configs.watchdog_config import *
from watchdog.gui.BaseWindow import BaseWindow

import numpy as np
import threading


class MainWindow(BaseWindow):
    def __init__(self, channel_pairs, screen=0, fullScreen=False, center=True, size=(None, None)):
        super().__init__(screen, fullScreen, center, size)
        self.channel_pairs = channel_pairs
        self.resultsList = []
        self.gridLayout = QGridLayout()
        self.wrapper = QVBoxLayout()
        self.resultListWrapper = QVBoxLayout()
        self.setLayout(self.gridLayout)

        # PLOT
        self.figure, self.axis = plt.subplots(2, 1, figsize=(16, 5))
        self.figure.tight_layout(h_pad=0.001)
        for ax in self.axis:
            ax.set_axis_off()
            ax.tick_params(labeltop=False, labelbottom=False, labelright=False, labelleft=False, bottom=False,
                           top=False, left=False, right=False)
        plt.subplots_adjust(wspace=0, hspace=0)

        self.canvas = FigureCanvas(self.figure)
        self.dataCorr = []
        self.dataBR = []
        self.vlines = []
        self.hlines = []
        self.initPlot()
        # PLOT

        # FORM
        self.HOST = QLineEdit()
        self.HOST.setText(HOST)

        self.PORT = QLineEdit()
        self.PORT.setText(str(PORT))

        self.rat_name = QLineEdit()
        self.rat_name.setText(rat_name)

        self.wait_time = QLineEdit()
        self.wait_time.setText(str(wait_time))

        self.epoch_time = QLineEdit()
        self.epoch_time.setText(str(epoch_time))

        self.sampling_rate = QLineEdit()
        self.sampling_rate.setText(str(sampling_rate))

        self.decimate_rate = QLineEdit()
        self.decimate_rate.setText(str(decimate_rate))

        self.num_channels = QLineEdit()
        self.num_channels.setText(str(num_channels))

        self.num_counter_for_refresh_animal = QLineEdit()
        self.num_counter_for_refresh_animal.setText(str(num_counter_for_refresh_animal))

        self.count_train_stimuls = QLineEdit()
        self.count_train_stimuls.setText(str(count_train_stimuls))

        self.train_step = QLineEdit()
        self.train_step.setText(str(train_step))

        self.data_source_is_file = QCheckBox()
        self.data_source_is_file.setChecked(data_source_is_file)

        self.is_train = QCheckBox()
        self.is_train.setChecked(is_train)

        self.start_button = QPushButton("Проверка")
        self.start_button.clicked.connect(self.__validate)
        self.stop_button = QPushButton("Стоп")

        self.formLayout = QFormLayout()
        self.formLayout.addRow("№ Объекта", self.rat_name)
        self.formLayout.addRow("Хост", self.HOST)
        self.formLayout.addRow("Порт", self.PORT)
        self.formLayout.addRow("Wait time", self.wait_time)
        self.formLayout.addRow("Epoch time", self.epoch_time)
        self.formLayout.addRow("S rate", self.sampling_rate)
        self.formLayout.addRow("D rate", self.decimate_rate)
        self.formLayout.addRow("Num channels", self.num_channels)
        self.formLayout.addRow("Limit", self.num_counter_for_refresh_animal)
        self.formLayout.addRow("Num train", self.count_train_stimuls)
        self.formLayout.addRow("Train step", self.train_step)
        self.formLayout.addRow("Is File", self.data_source_is_file)
        self.formLayout.addRow("Is Train", self.is_train)

        self.vbox = QVBoxLayout()
        for i in range(N):
            setattr(self, str(f"r{i}c1"), QLineEdit(odors[i][0]))
            setattr(self, str(f"r{i}c2"), QLineEdit(str(weights[i])))
            setattr(self, str(f"r{i}"), QtWidgets.QHBoxLayout())
            getattr(self, str(f"r{i}")).addWidget(getattr(self, str(f"r{i}c1")))
            getattr(self, str(f"r{i}")).addWidget(getattr(self, str(f"r{i}c2")))
            self.vbox.addLayout(getattr(self, str(f"r{i}")))

        self.formLayout.addRow(self.vbox)

        self.formLayout.addRow(self.start_button)
        self.formLayout.addRow(self.stop_button)
        # FORM

        self.accuracy = QVBoxLayout()
        self.accuracy_list = QtWidgets.QLineEdit()
        self.accuracy_list.setFixedHeight(50)
        self.accuracy.addWidget(self.accuracy_list)
        # ACCURACY

        font = QFont()
        font.setPointSize(50)

        self.resultListWidget = QtWidgets.QListWidget(self)
        self.resultListWidget.setFixedWidth(450)
        font = QFont()
        font.setPointSize(16)
        self.resultListWidget.setFont(font)
        self.resultListWrapper.addWidget(self.resultListWidget)
        self.gridLayout.addLayout(self.resultListWrapper, 1, Qt.AlignLeft)
        # ????
        self.gridLayout.addLayout(self.formLayout, 1, Qt.AlignTop)
        self.gridLayout.addLayout(self.accuracy, 0, Qt.AlignLeft)
        self.gridLayout.addWidget(self.canvas, 1, Qt.AlignRight)
        # ????
        # self.wrapper.addWidget(self.label, 1, Qt.AlignCenter)
        self.gridLayout.addLayout(self.wrapper, 0, 1, )

        self.fullScreen and self.showMaximized() or self.show()

    def showMessage(self, message, style="", timeout=0):
        '''
        Show message in main banner with specific style
        '''
        pass
        # if timeout:
        #     QTimer.singleShot(timeout, lambda: self.showMessage(message, style))
        #     return
        # self.label.setStyleSheet(style)
        # self.label.setText(message)

    def addResultListItem(self, text, bkgColor="grey"):
        '''
        Add item in resultListWidget
        '''
        item = QtWidgets.QListWidgetItem(text)
        item.setBackground(QColor(bkgColor))
        self.resultListWidget.addItem(item)
        self.resultListWidget.scrollToBottom()

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

    def showResult(self, message: str):
        QMessageBox.about(self, "Result", message)
        return self

    def show_acc(self):
        pass

    def __validate(self):
        current_setting = {
            "HOST": self.HOST.text(),
            "PORT": int(self.PORT.text()),
            "wait_time": int(self.wait_time.text()),
            "epoch_time": round(float(self.epoch_time.text()), 3),
            "sampling_rate": int(self.sampling_rate.text()),
            "decimate_rate": int(self.decimate_rate.text()),
            "num_channels": int(self.num_channels.text()),
            "num_counter_for_refresh_animal": int(self.num_counter_for_refresh_animal.text()),
            "count_train_stimuls": int(self.count_train_stimuls.text()),
            "train_step": int(self.train_step.text()),
            "data_source_is_file": bool(self.data_source_is_file.isChecked()),
            "is_result_validation": True,
            "is_train": bool(self.is_train.isChecked()),
            "use_auto_train": True,
            "odors": [(str(getattr(self, str(f"r{i}c1")).text()), "#ffff00") for i in range(N)],
            "odors_set": [2**i for i in range(N)],
            "weights": [float(getattr(self, str(f"r{i}c2")).text()) for i in range(N)],
            "unite": [[i] for i in range(N)],
            "unite_test": [[i] for i in range(N)],
            "rat_name": self.rat_name.text()
        }
        print(current_setting)
        Config(config_file, True, **current_setting)
        print("Replace!")
