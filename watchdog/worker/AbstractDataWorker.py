import threading

from PyQt5.QtCore import QThread, pyqtSignal
import random
import pandas as pd
from itertools import groupby

from classifier.kirilenko import kir_train_seq_SLP1
from classifier.kirilenko.kir_test_seq_SLP1 import *
from classifier.shepelev.ClassifierWrapper import ClassifierWrapper
from sklearn.metrics import confusion_matrix

class AbstractDataWorker(QThread):
    stopRecord = pyqtSignal(str, int)
    tick = pyqtSignal(object, float)
    startRecord = pyqtSignal()
    resultTest = pyqtSignal(str, int, object, int)
    resultTrain = pyqtSignal(int, int)
    tickViewSig = pyqtSignal(object)
    sendMessage = pyqtSignal(str)

    def __init__(self, bytes_to_read, decimate_rate, channel_pairs, path_to_res):
        super().__init__()
        self.classifierWrapper = ClassifierWrapper(num_channels=num_channels - 2, odors=odors_set, unite=unite)
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

    def predict(self, block):
        selected_classifiers = np.genfromtxt(os.path.join(out_path, "selected_classifiers.csv"), delimiter=",")
        resK = np.array([int(tst_Spl_original_signal(block))])
        resS = self.classifierWrapper.predict(np.array([np.transpose(block[prestimul_length:, :num_of_channels])]))
        res = np.concatenate((resS, resK), axis=None)
        res = self.classifierWrapper.convert_result_log(res)
        # Вывод только классификатор Шепелева
        return [self.get_result(np.array([res[int(i)] for i in selected_classifiers])), res]

    def get_result(self, res):
        x = pd.Series(res)
        x = x.value_counts()
        ind = x[x == x.max()].index
        if len(ind) > 1:
            if 1 in ind:
                return 1
            else:
                return 0
        else:
            return ind[0]

    def corrcoef_between_channels(self, data):
        return [abs(np.corrcoef((data[:, pair[0]], data[:, pair[1]]))[0][1]) for pair in
                np.asarray(self.channel_pairs)]

    def breathing_rate(self, sig):
        freq = np.fft.rfftfreq(sig.shape[0], 1. / sampling_rate)
        limit = 15
        return freq[(np.abs(np.fft.rfft(sig[:, num_channels - 2])) ** 2)[:len(freq[freq <= limit])].argmax(axis=0)]

    def train(self, path, stop=True):
        # self.sendMessage.emit("Начато обучение")
        # try:
        if stop:
            self.stop()
        res = self.classifierWrapper.train(path)
        res.append(kir_train_seq_SLP1.train(path))
        res1 = [(r[0], r[1]) for r in res]
        dat = np.fromfile(path, "i2").reshape(-1, num_channels)
        mask = np.isin(dat[:, -1], np.unique(dat[:, -1])[1:])
        labels = [dat[:, -1][mask][i] for i in range(dat[:, -1][mask].shape[0])]
        labels = [math.log(l, 2) for l in labels][-len(res[0][1]):]

        res = [(r[0], np.mean(np.array(self.classifierWrapper.convert_result_log(r[1])) == self.classifierWrapper.convert_result_log(labels)) * 100) for r in res]

        # for r in res1:
        #     print(confusion_matrix(self.classifierWrapper.convert_result_log(labels), self.classifierWrapper.convert_result_log(r[1])))

        np.savetxt(os.path.join(out_path, "acc_classifiers.csv"), np.array(res), delimiter=",")
        selected_classifiers = self.select(res)
        np.savetxt(os.path.join(out_path, "selected_classifiers.csv"), np.array(selected_classifiers), delimiter=",")

        return res
        # self.sendMessage.emit("Обучено")
        # except Exception:
        #     self.senMessage.emit("Ошибка при обучении")



    def validation_train(self, data):
        train_file_path = os.path.join(out_path, self.path_to_res + "_val.dat")
        np.copy(data).reshape(-1).astype('int16').tofile(train_file_path)

        self.create_inf(self.path_to_res + "_val", data.shape[0])

        # v = self.classifierWrapper.train(train_file_path)
        # v.append(kir_train_seq_SLP1.train(train_file_path))
        # # labels = np.concatenate([self.classifierWrapper.convert_result_log(odors_true) for i in range(round(len(v[1]) / len(odors_true)))])[-len(v[1]):]
        # dat = np.fprint_logromfile(train_file_path, "i2").reshape(-1, num_channels)
        # mask = np.isin(dat[:, -1], np.unique(dat[:, -1])[1:])
        # labels = [dat[:, -1][mask][i] for i in range(dat[:, -1][mask].shape[0])]
        # labels = [math.log(l, 2) for l in labels][-len(v[0][1]):]
        # v = [(r[0], np.mean(np.array(self.classifierWrapper.convert_result_log(r[1])) == labels) * 100) for r in v]
        # print(v)

        res = self.train(train_file_path, False)
        if np.mean(np.array([r[1] for r in res])) > validation_thresh:
            self.stop()
            self.train(os.path.join(train_file_path))

    def stop(self):
        self.working = False

    def select(self, val):
        # val - список или кортеж кортежей, где каждый внутренний кортеж соответствует одному классификатору и имеет
        # следующий вид: (*индекс классификатора*, *исходная точность валидации классификатора*)

        round_ = lambda x: int(5 * round(float(x) / 5))  # округление по нашему правилу
        val = [(clf[0], clf[1], round_(clf[1])) for clf in val]  # добавление округленной точности
        srt = sorted(val, key=lambda x: x[1], reverse=True)  # сортировка по не округленной точности
        td = list(
            filter(lambda x: x if x[2] > 50 else None, srt))  # отбрасывание плохих классификаторов (точность < 55 %)
        if len(td) != 0:
            if len(td) in (1, 2) and any(
                    [td_[1] >= 80 for td_ in td]):  # если остался один, причем очень хороший, оставляем только его
                return sorted([td_[0] for td_ in td])
            ranks = [(acc, list(clfs)) for (acc, clfs) in
                     groupby(td, lambda x: x[2])]  # ранжируем оставшиеся классификаторы по
            # округленным точностям
            res = [clf[0] for clf in ranks[0][1]]  # сразу заносим в итоговый результат классификаторы из первого ранга
            if len(res) >= 3:
                return sorted(res)
            else:
                for rank in ranks[1:]:  # идем от второго ранга к худшим
                    if rank[0] >= 70:  # если ранг не хуже 70%, берем все классификаторы в нем
                        res += [clf[0] for clf in rank[1]]
                    else:  # если ранг хуже 70%, добавляем из него по одному классификатору, пока не станет 3
                        for k in range(len(rank[1])):
                            res.append(rank[1][k][0])
                            if len(res) >= 3:
                                return sorted(res)
                    if len(res) >= 3:
                        return sorted(res)
        return sorted([srt[i][0] for i in range(3)])

    def create_inf(self, path_to_res, nNSamplings):
        with open(os.path.join(out_path, path_to_res + '.inf'), 'w') as f:
            f.write(
                "[Version]\nDevice=Smell-ADC\n\n[Object]\nFILE=\"\"\n\n[Format]\nType=binary\n\n[Parameters]\nNChannels={0}\nNSamplings={1}\nSamplingFrequency={2}\n\n[ChannelNames]\n{3}"
                .format(num_channels, nNSamplings, sampling_rate,
                        "\n".join(map(lambda x: str(x) + "=" + str(x + 1), range(num_channels)))))

    def dataProcessing(self, batch):
        self.tick.emit(self.corrcoef_between_channels(batch), self.breathing_rate(batch))

    def runThreadProcessing(self, batch):
        proc = threading.Thread(target=self.dataProcessing, args=[batch])
        proc.daemon = False
        proc.start()

    def runThreadTrain(self):
        proc = threading.Thread(target=self.train, args=[os.path.join(out_path, self.path_to_res + ".dat")])
        proc.daemon = False
        proc.start()

    def runThreadValidationTrain(self, data):
        proc = threading.Thread(target=self.validation_train, args=[data])
        proc.daemon = False
        proc.start()