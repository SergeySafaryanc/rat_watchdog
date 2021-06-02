from datetime import datetime
import threading

from PyQt5.QtCore import QThread, pyqtSignal
import pandas as pd
from itertools import groupby, permutations, product
import numpy as np
import os

from scipy.stats import entropy
from tqdm import tqdm

from classifier.kirilenko.KirClassifierWrapper import KirClassifierWrapper
from configs.watchdog_config import *
from classifier.shepelev.ClassifierWrapper import ClassifierWrapper

# from watchdog.utils.readme import readme
from loguru import logger
from itertools import chain

from watchdog.utils.readme import Singleton, write


class TimeInstance(object):
    """
    Singleton impl
    """
    time_now = datetime.now().strftime("%Y%m%d_%H_%M_%S")

    def __new__(cls):
        if not hasattr(cls, 'time_now'):
            cls.time_now = super(TimeInstance, cls).__new__(cls)
        return cls.time_now


class ExpFolder:
    def __init__(self):
        self.__time_now = TimeInstance()
        self.__exp_folder = f"{out_path}{os.sep}{self.__time_now}"
        if not is_train:
            folders = [fname for fname in os.listdir(f"{out_path}{os.sep}") if
                       os.path.isdir(os.path.join(f"{out_path}{os.sep}", fname))]
            # logger.info(folders)
            folders = sorted(folders, key=lambda x: os.stat(os.path.join(f"{out_path}{os.sep}", x)).st_mtime)
            logger.info(folders)
            self.__exp_folder = f"{out_path}{os.sep}{folders[-1]}"
            while (len(os.listdir(self.__exp_folder)) == 0):
                folders = folders[:-1]
                self.__exp_folder = f"{out_path}{os.sep}{folders[-1]}"
            logger.info(self.__exp_folder)
            return
        if os.path.exists(out_path) and os.path.exists(self.__exp_folder) is False:
            os.mkdir(self.__exp_folder)
            logger.debug(f"Folder - {self.__exp_folder} created")

    @property
    def time_now(self):
        return self.__time_now

    @property
    def exp_folder(self):
        return self.__exp_folder


class AbstractDataWorker(QThread, ExpFolder):
    stopRecord = pyqtSignal(str, int)
    tick = pyqtSignal(object, float)
    startRecord = pyqtSignal()
    resultTest = pyqtSignal(str, int, object, int)
    resultTrain = pyqtSignal(int, int)
    tickViewSig = pyqtSignal(object)
    sendMessage = pyqtSignal(str)

    def __init__(self, bytes_to_read, decimate_rate, channel_pairs, path_to_res, train_flag):
        super().__init__()

        # readme(is_first=True)

        # self.time_now = datetime.now().strftime("%Y%m%d_%H_%M_%S")
        # self.__exp_folder = f"{out_path}/{self.time_now}"
        # if os.path.exists(self.__exp_folder) is False:
        #     os.mkdir(self.__exp_folder)
        #     logger.debug(f"Folder - {self.__exp_folder} created")

        self.classifierWrapper = ClassifierWrapper(num_channels=num_channels - 2, odors=odors_unite, unite=unite)
        self.kirClassifierWrapper = KirClassifierWrapper()
        self.bytes_to_read = bytes_to_read
        self.counter = 0
        self.decimate_rate = decimate_rate
        self.sampling_rate = sampling_rate
        # индекс последней метки
        self.last_label_index = 0
        # индекс где была последняя корреляция
        self.last_corr_index = 0
        # вся запись
        self.record = np.array([])
        self.current_label = -1
        self.channel_pairs = channel_pairs
        self.path_to_res = path_to_res
        self.accuracy = []
        self.train_flag = train_flag
        if not data_source_is_file:
            self.path_to_res = self.path_to_res + "_" + self.time_now

        self.labels_map = {}  # новое автоматическое задание labels_map
        for i in list(chain(*odors_unite)):
            for j in range(len(odors_unite)):
                if i in odors_unite[j]:
                    self.labels_map[i] = j
        logger.info(f"labels_map: {self.labels_map}")

    def predict(self, block):
        selected_classifiers = np.genfromtxt(os.path.join(self.exp_folder, "selected_classifiers.csv"), delimiter=",")
        # selected_classifiers = np.genfromtxt(os.path.join(out_path, "selected_classifiers.csv"), delimiter=",")
        resK = self.kirClassifierWrapper.predict(block)
        resS = self.classifierWrapper.predict(np.array([np.transpose(block[prestimul_length:, :num_of_channels])]))
        logger.info(f"K: {str(resK)}")
        logger.info(f"S: {str(resS)}")
        res = np.concatenate((resS, resK), axis=None)
        logger.info(f"res before: {str(res)}")
        selected_classifiers = np.atleast_1d(selected_classifiers)  # костыль, когда один классификатор

        # result = self.get_result(np.array([res[int(i)] for i in selected_classifiers])) # старое получение результата

        res1 = np.atleast_2d(np.array([res[int(i)] for i in selected_classifiers]))  # преобразование ответов классификаторов
        logger.info(res1)
        result = self.test_by_clf_answers_weighted(res1, odors_groups_valtest)  # получение ответа комитета
        logger.info(result)

        res = self.classifierWrapper.convert_result_log(res)
        logger.info(f"res after: {res}")
        logger.info(f"selected_classifiers: {selected_classifiers}")

        return [result[0], res]

    # def predict_ter(self, block): # не используется
    #     selected_classifiers = np.genfromtxt(os.path.join(self.exp_folder, "selected_classifiers.csv"), delimiter=",")
    #     # selected_classifiers = np.genfromtxt(os.path.join(out_path, "selected_classifiers.csv"), delimiter=",")
    #     resK = self.kirClassifierWrapper.predict(block)
    #     resS = self.classifierWrapper.predict(np.array([np.transpose(block[prestimul_length:, :num_of_channels])]))
    #     print(f"K: {str(resK)}")
    #     print(f"S: {str(resS)}")
    #     res = np.concatenate((resS, resK), axis=None)
    #     print(f"res before: {str(res)}")
    #     selected_classifiers = np.atleast_1d(selected_classifiers)  # костыль, когда один классификатор
    #
    #     print(self.get_result(np.array([res[int(i)] for i in selected_classifiers])))
    #     print(res, 'before convert result')
    #     # Вывод только классификатор Шепелева
    #
    #     print(f"res after: {res}")
    #     print(f"selected_classifiers: {selected_classifiers}")
    #     print(res)
    #     print(selected_classifiers)
    #
    #     result_val = self.get_result(np.array([res[int(i)] for i in selected_classifiers]))  # получение одного ответа
    #     result_val = self.classifierWrapper.convert_result_log(np.atleast_1d(result_val))  # преобразование по группам
    #     result_val = result_val[0]  # преобразование в int
    #
    #     res = self.classifierWrapper.convert_result_log(res)  # преобразование всех ответов по группам
    #
    #     logger.info(result_val)
    #     logger.info(res)
    #
    #     return [result_val, res]

    def get_result_old(self, res): # устарело
        x = pd.Series(res)
        # logger.info(x)
        x = x.value_counts()
        # logger.info(x)
        ind = x[x == x.max()].index
        # logger.info(ind)
        if len(ind) > 1:
            if 1 in ind:
                return 1
            else:
                return 0
        else:
            return ind[0]

    def get_result(self, res):
        x = pd.Series(res)
        # logger.info(x)
        x = x.value_counts()
        # logger.info(x)
        x = pd.Series(data=map(lambda y, z: y * weights[z], x, x.index), index=x.index)  # числа ответов * веса ответов
        # logger.info(x)
        ind = x[x == x.max()].index
        # logger.info(ind)
        return ind[0]

    def corrcoef_between_channels(self, data):
        return [abs(np.corrcoef((data[:, pair[0]], data[:, pair[1]]))[0][1]) for pair in
                np.asarray(self.channel_pairs)]

    def breathing_rate(self, sig):
        freq = np.fft.rfftfreq(sig.shape[0], 1. / sampling_rate)
        limit = 15
        return freq[(np.abs(np.fft.rfft(sig[:, num_channels - 2])) ** 2)[:len(freq[freq <= limit])].argmax(axis=0)]

    def train(self, path, stop=True):
        if stop:
            self.stop()
        res = self.classifierWrapper.train(path)
        resK = self.kirClassifierWrapper.train(path)
        logger.info(f"AbstractDataWorker.py: res: {res}")
        logger.info(f"AbstractDataWorker.py: resK: {resK}")
        res = res + resK
        logger.info(res)
        res1 = [r[1] for r in res]
        logger.info(res1)
        val_res = res1  # создание переменной с ответами классификаторов для дальнейшего вывода в файл
        dat = np.fromfile(path, "i2").reshape(-1, num_channels)
        mask = np.isin(dat[:, -1], np.unique(dat[:, -1])[1:])
        # logger.info(mask)
        labels = [dat[:, -1][mask][i] for i in range(dat[:, -1][mask].shape[0])]
        logger.info(labels)
        # labels = [label for label in labels if label != 64] КОСТЫЛЬ НА УДАЛЕНИЕ 64 МЕТКИ
        # logger.info(labels)
        labels = [self.labels_map[l] for l in labels][-len(res[0][1]):]
        logger.info(labels)

        res = [(r[0], np.mean(
            np.array(self.classifierWrapper.convert_result_log(r[1])) == self.classifierWrapper.convert_result_log(
                labels)) * 100) for r in res]
        # logger.info(res)

        # logger.info(self.record.shape)
        # logger.info(f"AbstractDataWorker.py: res(convert_result_log): {[o[1] for o in res]}") # скрыли точности по клф
        logger.info(f"Точности в порядке сортировки: {sorted([o[1] for o in res], reverse=True)}")
        Singleton.set("Точность на валидации", f"{Singleton.get('Точность на валидации')}\n{res}")
        write(Singleton.text())

        np.savetxt(os.path.join(self.exp_folder, self.time_now + "_acc_classifiers.csv"), np.array(res),
                   delimiter=",")
        # np.savetxt(os.path.join(out_path, self.time_now + "_acc_classifiers.csv"), np.array(res), delimiter=",")
        selected_classifiers = self.select_ter(res)
        # logger.info(selected_classifiers)  #
        np.savetxt(str(os.path.join(self.exp_folder, f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}_selected_classifiers.csv")),
                   np.array(selected_classifiers),
                   # np.savetxt(os.path.join(out_path, self.time_now + "_selected_classifiers.csv"), np.array(selected_classifiers),
                   delimiter=",")
        np.savetxt(os.path.join(self.exp_folder, "selected_classifiers.csv"), np.array(selected_classifiers),
                   delimiter=",")
        # np.savetxt(os.path.join(out_path, "selected_classifiers.csv"), np.array(selected_classifiers), delimiter=",")

        # logger.info((np.array([res1[int(i)][r] for i in selected_classifiers])) for r in range(len(res1[0])))
        # answers = np.array( #раньше получали ответы тут
        #     [self.get_result(np.array([res1[int(i)][r] for i in selected_classifiers])) for r in range(len(res1[0]))])

        # ## теперь
        # res1 = np.transpose(np.array([res1[i] for i in selected_classifiers])) # преобразование ответов классификаторов
        #
        # answers = self.validate_by_clf_answers_weighted(res1,
        #     self.convert_result_group(labels, odors_groups_to_valtest), odors_groups_to_valtest)  # получение ответов комитета
        # # преобрование ответов комитета

        accuracy = 0  # задание низкой точности в случае невыбора классификаторов
        if len(selected_classifiers) >= 3:
            res1 = np.transpose(np.array([res1[i] for i in selected_classifiers]))  # преобразование ответов классификаторов
            logger.info(res1)

            weightsK = self.validate_by_clf_answers_weighted(res1,
                self.convert_result_group(labels, odors_groups_valtest),
                odors_groups_valtest)  # получение весов
            logger.info(weightsK)

            answers = list(self.test_by_clf_answers_weighted(res1, odors_groups_valtest))  # получение ответов комитета и преобразование
            logger.info(answers)
            # ## конец

            # logger.info(answers)  #
            val_res.append(np.array(answers))  # добавление ответов в вывод

            # logger.info(labels)
            val_res.append(np.array(labels))  # добавление реальных меток в вывод

            # logger.info(self.classifierWrapper.convert_result_log(answers))  #
            # logger.info(np.array(self.classifierWrapper.convert_result_log(answers)) == np.array(
            #     self.classifierWrapper.convert_result_log(labels)))  #
        
            accuracy = np.mean(np.array(self.classifierWrapper.convert_result_log(answers)) == np.array(
                self.classifierWrapper.convert_result_log(list(self.convert_result_group(labels, odors_groups_valtest))))) * 100

        np.savetxt(str(os.path.join(self.exp_folder,  f"{datetime.now().strftime('%Y%m%d_%H_%M_%S')}_valid_answers.csv")),
                   np.transpose(val_res),
                   fmt="%d")  # вывод ответов на валидации (классификаторы+реальные+полученные) в файл

        return accuracy

    def validation_train(self, data):
        train_file_path = os.path.join(self.exp_folder, self.path_to_res + "_val.dat")
        # train_file_path = os.path.join(out_path, self.path_to_res + "_val.dat")
        np.copy(data).reshape(-1).astype('int16').tofile(train_file_path)
        self.create_inf(self.path_to_res + "_val", data.shape[0])

        res = self.train(train_file_path, data_source_is_file)  # обучение и остановка на файле
        self.accuracy.append(res)
        with open(os.path.join(self.exp_folder, self.time_now + "_res.txt"), 'a+') as f:
            # with open(os.path.join(out_path, self.time_now + "_res.txt"), 'a+') as f:
            f.write(str(res))
            f.write('\n')

        if (self.accuracy[-1] >= acc_need or (
                len(self.accuracy) > 1 and (self.accuracy[-1] >= acc_need_two_ts) and (self.accuracy[-2] >= acc_need_two_ts))):

            # for k, v in Singleton.items():
            #     print(f"Key={k}\tValue={v}")

            self.stop()
            # self.sendMessage.emit("...")
            Singleton.set("Результат", "Обучено")
            write(Singleton.text())

            self.sendMessage.emit("Обучено")
            if one_file:
                self.applyTest()
                self.continueWork()
        if data_source_is_file and not (
                self.accuracy[-1] >= acc_need or (  # продолжение после остановки на файле, если НЕ обучено
                len(self.accuracy) > 1 and (self.accuracy[-1] >= acc_need_two_ts) and (self.accuracy[-2] >= acc_need_two_ts))):
            self.continueWork()

    def applyTest(self):
        self.train_flag = False

    def stop(self):  # остановка работы
        self.working = False

    def continueWork(self):  # продолжение работы
        self.working = True

    # def select(self, val):
    #     # val - список или кортеж кортежей, где каждый внутренний кортеж соответствует одному классификатору и имеет
    #     # следующий вид: (*индекс классификатора*, *исходная точность валидации классификатора*)
    #
    #     round_ = lambda x: int(5 * round(float(x) / 5))  # округление по нашему правилу
    #     val = [(clf[0], clf[1], round_(clf[1])) for clf in val]  # добавление округленной точности
    #     srt = sorted(val, key=lambda x: x[1], reverse=True)  # сортировка по не округленной точности
    #     td = list(
    #         filter(lambda x: x if x[2] > acc_need_of_one_clf else None, srt))  # отбрасывание плохих классификаторов
    #     if len(td) != 0:
    #         if len(td) in (1, 2) and any(
    #                 [td_[1] >= 80 for td_ in td]):  # если остался один (или 2), причем очень хороший, оставляем только его
    #             logger.info(srt)  #
    #             return sorted([td_[0] for td_ in td])
    #         ranks = [(acc, list(clfs)) for (acc, clfs) in
    #                  groupby(td, lambda x: x[2])]  # ранжируем оставшиеся классификаторы по
    #         # округленным точностям
    #         res = [clf[0] for clf in ranks[0][1]]  # сразу заносим в итоговый результат классификаторы из первого ранга
    #         if len(res) >= 3:
    #             return sorted(res)
    #         else:
    #             for rank in ranks[1:]:  # идем от второго ранга к худшим
    #                 if rank[0] >= 70:  # если ранг не хуже 70%, берем все классификаторы в нем
    #                     res += [clf[0] for clf in rank[1]]
    #                 else:  # если ранг хуже 70%, добавляем из него по одному классификатору, пока не станет 3
    #                     for k in range(len(rank[1])):
    #                         res.append(rank[1][k][0])
    #                         if len(res) >= 3:
    #                             logger.info(res)  #
    #                             return sorted(res)
    #                 if len(res) >= 3:
    #                     logger.info(res)  #
    #                     return sorted(res)
    #         logger.info(res)  #
    #         if len(res) >= 1:  # костыль для того, чтоб не возвращать хорошие с плохими (лучше 1-2, чем 3 с плохим)
    #             return sorted(res)  #
    #     logger.info(srt)  #
    #     return sorted([srt[i][0] for i in range(3)])

    def select(self, val):  # VERSION 2 of 03 March 2021
        # val - список или кортеж кортежей, где каждый внутренний кортеж соответствует одному классификатору и имеет
        # следующий вид: (*индекс классификатора*, *исходная точность валидации классификатора*)
        round_ = lambda x: int(5 * round(float(x) / 5))  # округление по нашему правилу
        val = [(clf[0], clf[1], round_(clf[1])) for clf in val]  # добавление округленной точности
        srt = sorted(val, key=lambda x: x[1], reverse=True)  # сортировка по не округленной точности
        td = list(
            filter(lambda x: x if x[1] >= acc_need_of_one_clf else None, srt))  # отбрасывание плохих классификаторов
        if len(td) >= 3:  # если после отбрасывания осталось 3 и больше
            ranks = [(acc, list(clfs)) for (acc, clfs) in
                     groupby(td, lambda x: x[2])]  # ранжируем оставшиеся классификаторы по округленным точностям
            # logger.info(ranks)  #
            res = []  # инициализируем массив результата
            # logger.info(res)  #
            for rank in ranks[0:]:  # набираем не менее 3 классификаторов по всем рангам последовательно по убыванию
                res += [clf[0] for clf in rank[1]]  # заносим в итоговый результат все классификаторы из ранга
                logger.info(res)  #
                if len(res) >= 3:  # если уже 3 и больше
                    return sorted(res)  # возвращаем
        else:  # возвращаем самый худший для того, чтоб не пройти валидацию на этом обучении и учиться дальше
            # logger.info(srt)  #
            srt = sorted(srt, key=lambda x: x[1])  # сортировка в порядке возрастания для возврата наименьшего
            # logger.info(srt)  #
            return [srt[0][0]]  # возвращаем индекс первого (худшего)

    def select_ter(self, val):
        selected = self.select(val)  # выбранные классификаторы
        # logger.info(selected)
        selected = list(map(lambda i: val[i], selected))  # по номерам получаем и номера, и точности
        # logger.info(selected)
        srt = sorted(selected,
                     key=lambda x: (-x[1], -x[0]))  # сортировка по убыванию точности и индекса (чтоб захватить Колины)
        # logger.info(srt)
        if (len(srt) > 3):  # если больше 3 вернул select
            # logger.info("len(srt)>3")
            for clf in srt[:3]:  # ищем LDA или SLP
                # logger.info(clf)
                if (clf[0]) == 7 or (clf[0]) == 6:  # если LDA или SLP
                    srt = srt[:4]  # берём ещё один классификатор
                    break
            else:
                srt = srt[:3]
            # logger.info(srt)
        return sorted([srt[i][0] for i in range(len(srt))])

    def create_inf(self, path_to_res, nNSamplings):
        with open(os.path.join(self.exp_folder, path_to_res + '.inf'), 'w') as f:
            # with open(os.path.join(out_path, path_to_res + '.inf'), 'w') as f:
            f.write(
                "[Version]\nDevice=Smell-ADC\n\n[Object]\nFILE=\"\"\n\n[Format]\nType=binary\n\n[Parameters]\nNChannels={0}\nNSamplings={1}\nSamplingFrequency={2}\n\n[ChannelNames]\n{3}"
                    .format(num_channels, nNSamplings, sampling_rate,
                            "\n".join(map(lambda x: str(x) + "=" + str(x + 1), range(num_channels)))))

    # def correct_labels_by_groups(self, data):
    #     for group in odors_unite:  # просматриваем в цикле группы
    #         if len(group) == 1:  # если число элементов в группе равно одному
    #             continue  # пропускаем
    #         for i in range(1, len(group)):  # для каждого индекса группы, кроме первого
    #             data[:, -1] = np.where(data[:, -1] == group[i], group[0], data[:, -1])  # замена метки на первую в группе
    #     # for i in range(data.shape[0]):  # для всех меток клапанов
    #     #     if data[i, -1] != 0:  # если метка не нулевая
    #     #         for group in odors_unite:  # просматриваем в цикле группы
    #     #             if data[i, -1] in group:  # если принадлежит группе
    #     #                 data[i, -1] = group[0]  # присваиваем метку первого элемента группы
    #     return data

    def clf_answers_to_result_weighted(self, current_answer_array, grouping_map, weight_dict):
        """
        current_answer_array - одномерный массив ответов лучших (зачастую трех) классификторов
                current_answer_array=np.asarray([0,1,4])
        grouping_map - группировка в формате [[1,4][3,5][2]]
        weight_dict- словарь весов для клапанов в формате:
        weight_dict = np.asarray([[0,0.99],
                                  [1,0.99],
                                  [2,0.1],
                                  [3,0.99],
                                  [4,0.4],
                                  [5,0.2],
                                  [6,0.2],
                                  [7,0.2]]).astype('float64')
        """
        clf_answers_weights = []
        for ans_it in range(current_answer_array.shape[0]):
            for dict_i in range(weight_dict.shape[0]):
                if current_answer_array[ans_it] == weight_dict[dict_i, 0]:
                    clf_answers_weights.append(weight_dict[dict_i, 1])
        clf_answers_weights = np.asarray(clf_answers_weights)
        # сортировки и сравнения производятся по ответм и грппам, но все вычисления делаются по весам
        # Для проверки оставлены участки кода с вычислениям аналогичным по ответам, чтобы сопоставить правильность
        # print(current_answer)
        # print(clf_answers_weights)
        # ans_by_grop_list = []  #не удалять, нужно для проверки в будущем
        weight_by_grop_list = []
        for group_i in range(len(grouping_map)):
            # ans_by_grop = []  #не удалять, нужно для проверки
            weight_by_grop = []
            for ans_i in range(current_answer_array.shape[0]):
                if current_answer_array[ans_i] in grouping_map[group_i]:
                    # ans_by_grop.append(current_answer[ans_i])  #не удалять, нужно для проверки в будущем
                    weight_by_grop.append(clf_answers_weights[ans_i])
                else:
                    # ans_by_grop.append(nan) #не удалять, нужно для проверки в будущем
                    weight_by_grop.append(0)
            # ans_by_grop= np.asarray(ans_by_grop) #не удалять, нужно для проверки в будущем
            weight_by_grop = np.asarray(weight_by_grop)
            # ans_by_grop_list.append(ans_by_grop) #не удалять, нужно для проверки в будущем
            weight_by_grop_list.append(weight_by_grop)

        result_weight = []
        for gr_it in range(len(weight_by_grop_list)):
            result_weight.append(sum(weight_by_grop_list[gr_it]))
        result_ind = np.argmax(np.asarray(result_weight))
        return result_ind

    def validate_by_clf_answers_weighted(self, clf_answers, real_answers, grouping_map):
        """
        clf_answers - ответы  лучших классификаторов(зачастую трех) по отдельным калапанам. т.е. если 5 клапанов то диапазон ответов [0,1,2,3,4]
                        ответы подавать в виде 2d -массива, где строки - предъявления(стимулы), а столбцы - номер классификатора
                        clf_answers = np.vstack([[0,1,2],
                                                 [1,2,2],
                                                 [2,2,2],
                                                 [3,4,3],
                                                 [2,4,4],
                                                 [0,0,1],
                                                 [2,4,2]])
        real_answers- реальные метки ответов в виде индексов групп, т.е. если были клапаны 1,2,3,4,5, которые группируются [[1,4][3,5][2]], то
                        индекс группы   0       1     2
                        группа        [0,1]   [3,5]  [2]

            real_answers = np.asarray([0,0,0,1,1,0,0])

        grouping_map - группировка в формате [[1,4][3,5][2]]    
        """
        all_clapans = sum(grouping_map, [])
        all_clapans.sort()
        weight_shift = 0.1
        weight_array = np.arange(0.4, 1.2, weight_shift)
        weight_combinations = [p for p in product(weight_array, repeat=len(all_clapans))]
        acc_list = []
        comb_list = []
        comb_ans = []
        for i in tqdm(range(len(weight_combinations))):
            comb = np.asarray(weight_combinations[i])
            cur_weight_dict = np.vstack([all_clapans, comb]).T
            comb_result = []
            for j in range(clf_answers.shape[0]):
                res = self.clf_answers_to_result_weighted(clf_answers[j], grouping_map, weight_dict=cur_weight_dict)
                comb_result.append(res)
            num_of_true = 0
            comb_ans.append(np.asarray(comb_result))
            for i in range(len(comb_result)):
                if comb_result[i] == real_answers[i]:
                    num_of_true += 1
            acc = (num_of_true / len(comb_result)) * 100
            # print(comb,'=========>',acc)
            acc_list.append(acc)
            comb_list.append(comb)
        comb_list - np.vstack(comb_list)
        acc_list = np.asarray(acc_list)
        comb_ans = np.vstack(comb_ans)
        comb_list = np.vstack(comb_list)
        all_best_comb = np.where(acc_list == np.amax(acc_list))[0]
        best_comb_list = comb_list[all_best_comb]
        best_ans_comb_list = comb_ans[all_best_comb]
        list_of_entr_for_best_comb = []
        err_in_group_for_comb = []
        for ans in range(best_ans_comb_list.shape[0]):
            cur_best_ans = best_ans_comb_list[0]
            err_arr = []
            for a in range(cur_best_ans.shape[0]):
                if cur_best_ans[a] != real_answers[a]:
                    err_arr.append(real_answers[a])
            err_arr = np.asarray(err_arr)
            err_in_group = []
            for gr in range(len(grouping_map)):
                num_of_err_in_gr = 0
                for err in range(err_arr.shape[0]):
                    if err_arr[err] in grouping_map[gr]:
                        num_of_err_in_gr += 1
                err_in_group.append(num_of_err_in_gr)
            err_in_group = np.asarray(err_in_group)
            err_in_group_for_comb.append(err_in_group)
            sum_of_err = sum(err_in_group)
            prob_of_err = (err_in_group + 1) / sum_of_err
            enntrop_for_ans = entropy(prob_of_err)
            list_of_entr_for_best_comb.append(enntrop_for_ans)
        list_of_entr_for_best_comb = np.asarray(list_of_entr_for_best_comb)
        err_in_group_for_comb = np.vstack(err_in_group_for_comb)

        best_comb_by_entr_id = np.where(list_of_entr_for_best_comb == np.amax(list_of_entr_for_best_comb))[0]
        best_comb_by_entr = best_comb_list[best_comb_by_entr_id]
        super_best_comb = best_comb_by_entr[0]
        np.savetxt(fname='super_best_weights.weight',X=super_best_comb, delimiter=';')
        return super_best_comb

        # return comb_list,acc_list,comb_ans
        # all_best_comb = np.where(acc_list==np.amax(acc_list))[0]
        # super_best_weights = np.mean(np.vstack(comb_list)[all_best_comb], axis=0)
        # np.savetxt(fname='super_best_weights.weight',X=super_best_weights, delimiter=';')
        # return super_best_weights, np.amax(acc_list), comb_ans

    def test_by_clf_answers_weighted(self, clf_answers, grouping_map):
        """
        clf_answers - ответы  лучших классификаторов(зачастую трех) по отдельным калапанам. т.е. если 5 клапанов то диапазон ответов [0,1,2,3,4]
                        ответы подавать в виде 2d -массива, где строки - предъявления(стимулы), а столбцы - номер классификатора
                        clf_answers = np.vstack([[0,1,2],
                                                 [1,2,2],
                                                 [2,2,2],
                                                 [3,4,3],
                                                 [2,4,4],
                                                 [0,0,1],
                                                 [2,4,2]])
        clf_clapan_weights = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9]) для клапанов 0,1,2,3,4

        grouping_map - группировка в формате [[0,3][2,4][1]]    
        """
        clf_clapan_weights = np.genfromtxt(fname='super_best_weights.weight', delimiter=';')
        all_clapans = sum(grouping_map, [])
        all_clapans.sort()
        weight_dict = np.vstack([all_clapans, clf_clapan_weights]).T
        result = []
        for j in range(clf_answers.shape[0]):
            res = self.clf_answers_to_result_weighted(clf_answers[j], grouping_map, weight_dict=weight_dict)
            result.append(res)
        return np.asarray(result)

    def convert_units_to_groups(self, val_real_ans_by_clapan_id, grouping_map):
        ans_by_group = []
        for ans_it in range(val_real_ans_by_clapan_id.shape[0]):
            ans = val_real_ans_by_clapan_id[ans_it]
            for group in range(len(grouping_map)):
                if ans in grouping_map[group]:
                    converted_ans = group
                    ans_by_group.append(converted_ans)
        ans_by_group = np.asarray(ans_by_group)
        return ans_by_group

    def convert_result_group(self, res, groups):
        result = []
        for i in range(len(res)):
            for j in range(len(groups)):
                if res[i] in groups[j]:
                    result.append(j)
        return np.asarray(result)


    def dataProcessing(self, batch):
        self.tick.emit(self.corrcoef_between_channels(batch), self.breathing_rate(batch))

    def runThreadProcessing(self, batch):
        proc = threading.Thread(target=self.dataProcessing, args=[batch])
        proc.daemon = False
        proc.start()

    def runThreadTrain(self):
        # proc = threading.Thread(target=self.train, args=[os.path.join(out_path, self.path_to_res + ".dat")])
        proc = threading.Thread(target=self.train, args=[os.path.join(self.exp_folder, self.path_to_res + ".dat")])
        proc.daemon = False
        proc.start()

    def runThreadValidationTrain(self, data):
        proc = threading.Thread(target=self.validation_train, args=[data])
        proc.daemon = False
        proc.start()
