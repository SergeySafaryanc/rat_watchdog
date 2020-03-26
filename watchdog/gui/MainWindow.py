from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QGridLayout
from PyQt5.QtWidgets import QVBoxLayout
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QTimer

from configs.watchdog_config import is_train
from watchdog.gui.BaseWindow import BaseWindow


class MainWindow(BaseWindow):
    def __init__(self, screen=0, fullScreen=False, center=True, size=(None, None)):
        super().__init__(screen, fullScreen, center, size)
        self.resultsList = []
        self.gridLayout = QGridLayout()
        self.wrapper = QVBoxLayout()
        self.resultListWrapper = QVBoxLayout()
        self.setLayout(self.gridLayout)

        self.label = QtWidgets.QLabel(self)
        self.label.setMinimumWidth(900)
        self.label.setMinimumHeight(200)
        font = QFont()
        font.setPointSize(50)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        if is_train:
            self.trainButton = QtWidgets.QPushButton('Train', self)
            self.resultListWrapper.addWidget(self.trainButton)

        self.resultListWidget = QtWidgets.QListWidget(self)
        self.resultListWidget.setFixedWidth(450)
        font = QFont()
        font.setPointSize(16)
        self.resultListWidget.setFont(font)
        self.resultListWrapper.addWidget(self.resultListWidget)
        self.gridLayout.addLayout(self.resultListWrapper, 0, 0)
        self.wrapper.addWidget(self.label, 1, Qt.AlignCenter)
        self.gridLayout.addLayout(self.wrapper, 0, 1, )

        self.fullScreen and self.showMaximized() or self.show()

    def showMessage(self, message, style="", timeout=0):
        '''
        Show message in main banner with specific style
        '''
        if timeout:
            QTimer.singleShot(timeout, lambda: self.showMessage(message, style))
            return
        self.label.setStyleSheet(style)
        self.label.setText(message)

    def addResultListItem(self, text, bkgColor="grey"):
        '''
        Add item in resultListWidget
        '''
        item = QtWidgets.QListWidgetItem(text)
        item.setBackground(QColor(bkgColor))
        self.resultListWidget.addItem(item)
        self.resultListWidget.scrollToBottom()
