from PyQt5.QtCore import QThread, pyqtSignal
from time import sleep

from classifier.kirilenko.kir_test_seq_SLP1 import *


class MainWorker(QThread):
    filesDetected = pyqtSignal(set)

    def __init__(self, path, wait_time):
        super().__init__()
        self.path = path
        self.wait_time = wait_time

    def run(self):
        files_list = set(os.listdir(self.path))
        print('Waiting for data.')
        while True:
            new_list = set(os.listdir(self.path))
            if new_list != files_list:
                diff = new_list - files_list
                files_list = new_list
                if len(diff) > 0:
                    print('%i new file(s) detected.' % len(diff))
                    self.filesDetected.emit(diff)
                    print('Waiting for data.')
            sleep(self.wait_time)
