import os

from watchdog.utils import vk_bot
from watchdog.worker.AbstractDataWorker import ExpFolder
from watchdog.worker.AbstractDataWorker import AbstractDataWorker
from watchdog.worker.FileDataWorker import FileDataWorker
from watchdog.worker.MainWorker import MainWorker
from watchdog.gui.GUI import GUI
import numpy as np
from itertools import combinations
from configs.watchdog_config import *
from watchdog.utils.readme import Singleton
from watchdog.worker.SocketDataWorker import SocketDataWorker
from loguru import logger

from random import random


class QtWatchdogController(ExpFolder):
    def __init__(self):
        Singleton.set("Крыса", rat_name)
        Singleton.set("Каналов", num_channels)
        Singleton.set("Метки веществ", dict(zip([o[0] for o in odors_unite], [i[0] for i in odors])))

        super().__init__()
        self.channel_pairs = list(combinations(np.arange(0, num_of_channels), 2))
        self.gui = GUI(channel_pairs=self.channel_pairs)
        self.workers = {}
        self.resultsCounter = 0
        self.isRecording = False
        self.rec = []
        self.dataBR = []
        self.dataCorr = []
        self.validationResult = [validationResult,
                                 validationResult]
        if os.path.exists(os.path.join(self.exp_folder, 'correlations.npy')):
            self.dataCorr = np.load(os.path.join(self.exp_folder, 'correlations.npy')).tolist()
        if os.path.exists(os.path.join(self.exp_folder, 'breathing_rate.npy')):
            self.dataBR = np.load(os.path.join(self.exp_folder, 'breathing_rate.npy')).tolist()

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
        np.save(os.path.join(self.exp_folder, 'correlations'), dataCorrNP)
        # np.save(os.path.join(out_path, 'correlations'), dataCorrNP)
        np.save(os.path.join(self.exp_folder, 'breathing_rate'), dataBRNP)
        # np.save(os.path.join(out_path, 'breathing_rate'), dataBRNP)

    def onTickViewSig(self, sig):
        self.gui.plotWindow.addPointSig(sig)

    def onResultTest(self, name, i, results, label, odors_, labels_missing_counter=0):
        for lm in range(labels_missing_counter):
            self.resultsCounter += 1

            message, color = "%i. %s" % (self.resultsCounter, result_messages[1][0]), result_messages[1][1]

            self.gui.mainWindow.addResultListItem(message, color)

            # вывод пропуска в логах
            with open(os.path.join(self.exp_folder, name + '_responses_classifiers_and_result.csv'), 'a+') as f:
                f.write(
                    ';'.join([str(self.resultsCounter), ";".join(list(result_messages[1][0] for i in range(10)))]))
                f.write('\n')

        print(results)
        print(label)
        print(odors_)
        self.resultsCounter += 1
        result, resСlassifiers = results

        message, color = odors_[int(result)]

        # with open(os.path.join(self.exp_folder, name + '_result.csv'), 'a+') as f:
        # # with open(os.path.join(out_path, name + '_result.csv'), 'a+') as f:
        #     f.write(';'.join([str(self.resultsCounter), name + '_' + str(i), message]))
        #     f.write('\n')
        # with open(os.path.join(self.exp_folder, name + '_result_labels.csv'), 'a+') as f:
        # # with open(os.path.join(out_path, name + '_result.csv'), 'a+') as f:
        #     f.write(';'.join([str(self.resultsCounter), name + '_' + str(i), str(result)]))
        #     f.write('\n')

        # НИКАКИХ ГРУПП НА ВАЛИДАЦИИ!!
        # # вывод ответов по всем классификаторам и предикта по комитету в виде меток
        # with open(os.path.join(self.exp_folder, name + '_responses_classifiers_and_result_labels.csv'), 'a+') as f:
        #     f.write(';'.join([str(self.resultsCounter), ";".join(map(lambda x: str(x), resСlassifiers)), str(result),
        #                       str(label)]))
        #     f.write('\n')
        #
        # # преобразование ответов классификаторов
        # logger.info(resСlassifiers)
        # resСlassifiers = self.convert_result_group(resСlassifiers, odors_groups_valtest)
        # logger.info(resСlassifiers)
        # # преобразование ответа комитета
        # logger.info(label)
        # label = self.convert_result_group(np.atleast_1d(np.asarray(label)), odors_groups_valtest)[0]
        # logger.info(label)
        # # преобразование общего массива
        # results = [result, resСlassifiers]

        # вывод ответов по всем классификаторам и предикта по комитету в текстовом виде
        with open(os.path.join(self.exp_folder, name + '_responses_classifiers_and_result.csv'), 'a+') as f:
            # with open(os.path.join(out_path, name + '_responses_classifiers.csv'), 'a+') as f:
            f.write(';'.join([str(self.resultsCounter), ";".join(map(lambda x: odors_[x][0], resСlassifiers)), message,
                              odors_[int(label)][0]]))
            f.write('\n')

        message = "%i. %s" % (self.resultsCounter, message)

        if is_result_validation:
            color = self.resultValidation(results, label, name)

        self.gui.mainWindow.showMessage(message, "background: %s" % color)
        self.gui.mainWindow.addResultListItem(message, color)

        message, color = result_messages[0]

        self.gui.mainWindow.showMessage(message, "background: %s" % color, show_result_delay * 1000)

    def onResultTestTrash(self, name, i, results, label):
        print(f"results: {results}")
        print(f"label: {label}")

        # if results[0][0] == 0:
        if label == 0:
            result, resСlassifiers = results[1]
            _odors = odors_2[int(result)]
            message, color = _odors
            odv = odors_groups_valtest_2
            label = result
        else:
            result, resСlassifiers = results[0]
            _odors = odors[int(result)]
            message, color = _odors
            odv = odors_groups_valtest
            label = 1

        # result, resСlassifiers = results[0] if results[0][0] != 1 else results[1]

        # breakpoint()
        # print(o)
        self.resultsCounter += 1
        # result, resСlassifiers = results if label1 == 8887232 else results1
        # message, color = odors[int(result)] if label1 == 8887232 else odors_2[int(result)]
        # odv = odors_groups_valtest if label1 == 8887232 else odors_groups_valtest_2

        print(f"""
        
            result, resСlassifiers := {str(result)}, {str(resСlassifiers)}\n
            message, color := {str(message)}, {str(color)}\n
            odv := {str(odv)}
        """)

        # selected_index = 1 if label == 0 else 0
        # result, resClassifiers = o[selected_index]["results"]
        # message, color = o[selected_index]["odors"][label1 if label1 != 8887232 else label]
        # message, color = o[selected_index]["odors"][label1 if label1 != 8887232 else label]
        # label = o[selected_index]["label"]
        # message, color = o[selected_index]["odors"][label]
        # odv = o[selected_index]["odv"]
        # print(message)
        # breakpoint()



        # result, resСlassifiers = results
        # message, color = odors[int(result)]
        # если на message == ЦВ
        # if message == odors[0][0]:
        #     result, resСlassifiers = results1
        #     message, color = odors_2[int(result)]
        # print(f"Вывод на экран: {message}")

        # вывод ответов по всем классификаторам и предикта по комитету в виде меток
        # with open(os.path.join(self.exp_folder, name + '_0_responses_classifiers_and_result_labels.csv'), 'a+') as f:
        #     f.write(';'.join([str(self.resultsCounter), ";".join(map(lambda x: str(x), resСlassifiers)), str(result),
        #                       str(label)]))
        #     f.write('\n')

        # logger.info(resСlassifiers)
        resClassifiers = self.convert_result_group(resСlassifiers, odv)
        # logger.info(resСlassifiers)
        # преобразование ответа комитета
        # logger.info(label)
        label = self.convert_result_group(np.atleast_1d(np.asarray(label)), odv)[0]
        # logger.info(label)
        # преобразование общего массива
        results = [result, resClassifiers]
        # print(label)
        # print(o[selected_index]["odors"])
        # print(o[selected_index]["odors"][int(label)][0])
        # print(o[selected_index]["odors"])
        # print(list(map(lambda x: o[selected_index]["odors"][x][0], resClassifiers)))
        # breakpoint()
        # вывод ответов по всем классификаторам и предикта по комитету в текстовом виде
        with open(os.path.join(self.exp_folder, name + '_0_responses_classifiers_and_result.csv'), 'a+') as f:
            f.write(';'.join([str(self.resultsCounter),
                              ";".join(list(map(lambda x: _odors[x][0], resClassifiers))),
                              message,
                              str(_odors[int(label)][0])]
                             ))
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

    def resultValidation(self, results, label, name):
        result, resultList = results
        errorColor = "#cccccc"  # "#db3b21"
        okColor = "#cccccc"  # "#60f750"
        color = (okColor if result == label else errorColor)
        print()
        self.validationResult[0][result] += 1 if result == label else 0
        self.validationResult[1][resultList[-1]] += 1 if resultList[-1] == label else 0
        logger.info(f"QtWatchdogController.validationResult: {self.validationResult[0].keys()}")
        logger.info(f"Odors: {odors}")
        with open(os.path.join(self.exp_folder, name + '_validation_result.csv'), 'w') as f:
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
            logger.info("file %s" % f)
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

    def convert_result_group(self, res, groups):  # конвертация для ВАЛИДАЦИИ ПО ГРУППАМ для выбора весов
        result = []
        for i in range(len(res)):
            for j in range(len(groups)):
                if res[i] in groups[j]:
                    result.append(j)
        return np.asarray(result)

    def get_label_and_resClass(self, rc, label, odv):
        resСlassifiers = self.convert_result_group(rc, odv)
        label = self.convert_result_group(np.atleast_1d(np.asarray(label)), odv)[0]
        return label, resСlassifiers

    # def select_odor(self, odors: list, results: list, labels: list, results1, label1):
    # def select_odor(self, o: list, default_odor=odors[0][0]):
    #     """
    #     o = [{
    #         "odors": (str, str),
    #         "results": list[int, list[int]],
    #         "label": int,
    #         "odv": list[]
    #     }]
    #     """
    #
    #     print(f"Вывод на экран: {message}")
    #     return dict(
    #         result=result,
    #         message=message,
    #         resClassifiers=self.convert_result_group(res_classifiers, odv),
    #         label=self.convert_result_group(np.atleast_1d(np.asarray(label)), odv)[0]
    #     )
