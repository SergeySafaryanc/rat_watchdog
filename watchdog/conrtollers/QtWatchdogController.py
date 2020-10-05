import os

from watchdog.utils import vk_bot
from watchdog.worker.FileDataWorker import FileDataWorker
from watchdog.worker.MainWorker import MainWorker
from watchdog.gui.GUI import GUI
from configs.watchdog_config import *
import numpy as np
from itertools import combinations

from watchdog.worker.SocketDataWorker import SocketDataWorker
from loguru import logger


class QtWatchdogController:
    def __init__(self):
        self.channel_pairs = list(combinations(np.arange(0, num_of_channels), 2))
        self.gui = GUI(channel_pairs=self.channel_pairs)
        self.workers = {}
        self.resultsCounter = 0
        self.isRecording = False
        self.rec = []
        self.dataBR = []
        self.dataCorr = []
        self.validationResult = [{0: 0, 1: 0, 2: 0}, {0: 0, 1: 0, 2: 0}]#[{0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}]
        if os.path.exists(os.path.join(out_path, 'correlations.npy')):
            self.dataCorr = np.load(os.path.join(out_path, 'correlations.npy')).tolist()
        if os.path.exists(os.path.join(out_path, 'breathing_rate.npy')):
            self.dataBR = np.load(os.path.join(out_path, 'breathing_rate.npy')).tolist()

        if data_source_is_file:
            worker = MainWorker(inp_path, wait_time)
            worker.filesDetected.connect(self.onFilesDetected)
            worker.start()
            self.workers['main'] = worker
        else:
            # 2 - размер short
            # 1 - АЦП
            worker = SocketDataWorker(2 * sampling_rate * (num_channels + 1) * decimate_rate, decimate_rate,
                                      num_channels + 1, self.channel_pairs, is_train)
            self.initDataWorker(worker)
            if is_train:
                self.gui.mainWindow.trainButton.clicked.connect(worker.runThreadTrain)
            worker.start()
            self.workers['main_thread'] = worker

        self.gui.start()

    def onTick(self, corr, br):
        self.dataCorr.append(corr)
        self.dataBR.append(br)

        dataCorrNP = np.array(self.dataCorr)
        dataBRNP = np.array(self.dataBR)
        self.gui.plotWindow.addPoint(dataCorrNP, dataBRNP)
        np.save(os.path.join(out_path, 'correlations'), dataCorrNP)
        np.save(os.path.join(out_path, 'breathing_rate'), dataBRNP)

    def onTickViewSig(self, sig):
        self.gui.plotWindow.addPointSig(sig)

    def onResultTest(self, name, i, results, label):
        self.resultsCounter += 1
        result, resСlassifiers = results
        message, color = odors[int(result)]

        with open(os.path.join(out_path, name + '_result.csv'), 'a+') as f:
            f.write(';'.join([str(self.resultsCounter), name + '_' + str(i), message]))
            f.write('\n')

        with open(os.path.join(out_path, name + '_responses_classifiers.csv'), 'a+') as f:
            f.write(';'.join([str(self.resultsCounter), ";".join(map(lambda x: odors[x][0], resСlassifiers))]))
            f.write('\n')

        message = "%i. %s" % (self.resultsCounter, message)

        if is_result_validation:
            color = self.resultValidation(results, label, name)

        self.gui.mainWindow.showMessage(message, "background: %s" % color)
        self.gui.mainWindow.addResultListItem(message, color)

        message, color = result_messages[0]

        self.gui.mainWindow.showMessage(message, "background: %s" % color, show_result_delay * 1000)

    def onResultTrain(self, i, label):
        self.resultsCounter += 1

        if self.workers['main_thread'] == True:
            message = "%i. %s" % (self.resultsCounter, label)

            self.gui.mainWindow.showMessage(message, "background: %s" % result_messages[0][1])
            self.gui.mainWindow.addResultListItem(message, result_messages[0][1])

            message, color = result_messages[0]

            self.gui.mainWindow.showMessage(message, "background: %s" % color, show_result_delay * 1000)
        else:
            message = "%i. %s" % (self.resultsCounter, label)

            self.gui.mainWindow.showMessage(message, "background: %s" % result_messages[0][1])
            self.gui.mainWindow.addResultListItem(message, result_messages[0][1])

    def sendMessage(self, message):
        self.gui.mainWindow.showMessage(message, "background: #CCC")
        # try:
        #     vk_bot.send_message(message)
        # except Exception as e:
        #     print(e)

    def resultValidation(self, results, label, name):
        result, resultList = results
        errorColor = "#db3b21"
        okColor = "#60f750"
        color = (okColor if result == label else errorColor)
        print ()
        self.validationResult[0][result] += 1 if result == label else 0
        self.validationResult[1][resultList[-1]] += 1 if resultList[-1] == label else 0
        print(f"QtWatchdogController.validationResult: {self.validationResult[0].keys()}")
        print(f"Odors: {odors}")
        with open(os.path.join(out_path, name + '_validation_result.csv'), 'w') as f:

            f.write(';'.join(map(lambda x: odors[x][0], self.validationResult[0].keys())))
            f.write('\n')
            f.write(';'.join(map(str, self.validationResult[0].values())))
            f.write('\n')
            f.write(';'.join(map(str, self.validationResult[1].values())))
            f.write('\n')
        return color

    def onFilesDetected(self, files_list):
        if not is_train:
            assert len(files_list) == 1, "Only one file can be processed, sorry!"
        for f in files_list:
            print("file %s" % f)
            key, ext = os.path.splitext(f)
            if ext == '.inf':
                if key in self.workers:
                    w = self.workers[key]
                    w.stop()
                    del self.workers[key]
                continue
            # 2 - размер short
            worker = FileDataWorker(os.path.join(inp_path, f), key, 2 * sampling_rate * num_channels, epoch_time,
                                    decimate_rate, self.channel_pairs, is_train)
            self.initDataWorker(worker)
            worker.start()
            self.workers['main_thread'] = worker
            # worker.train(os.path.join(inp_path, f))
            message, color = result_messages[0]
            self.gui.mainWindow.showMessage(message, "background: %s" % color)
            self.workers[key] = worker

    def initDataWorker(self, worker):
        worker.tick.connect(self.onTick)
        worker.tickViewSig.connect(self.onTickViewSig)
        worker.resultTest.connect(self.onResultTest)
        worker.resultTrain.connect(self.onResultTrain)
        worker.sendMessage.connect(self.sendMessage)
