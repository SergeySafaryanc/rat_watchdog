import math
import os

from classifier.kirilenko import kir_train_seq_SLP1
from configs.watchdog_config import *
from watchdog.worker.AbstractDataWorker import AbstractDataWorker
from time import sleep
import numpy as np


class FileDataWorker(AbstractDataWorker):
    def __init__(self, f_name, name, bytes_to_read, epoch_time, decimate_rate, channel_pairs, train_flag):
        super().__init__(bytes_to_read, decimate_rate, channel_pairs, name, train_flag)
        self.working = True
        self.f_name = f_name
        self.name = name
        self.is_test_started = True
        self.epoch_time = epoch_time
        self.label_index_list = []

    def run(self):
        find_label = True
        with open(self.f_name, 'rb', buffering=0) as f:

            while True:
                if not self.working:
                    sleep(10)
                    continue
                sleep(self.epoch_time)
                bytes = f.read(self.bytes_to_read)

                if len(bytes) < self.bytes_to_read:
                    f.seek(-len(bytes), 1)
                    continue
                self.record = np.append(self.record, np.frombuffer(bytes, 'i2'))

                data = self.record.reshape((-1, num_channels))

                if find_label:
                    for i in reversed(range(data.shape[0] - sampling_rate, data.shape[0] - 1)):
                        if i > self.last_label_index and (data[i, -1] not in (0, 64)):
                            self.last_label_index = i
                            label = data[i, -1]
                            find_label = False
                            break

                if find_label or data[self.last_label_index:].shape[0] < clapan_length:
                    if data[:self.last_label_index].shape[0] < prestimul_length:
                        find_label = True
                    continue
                else:
                    find_label = True

                self.label_index_list.append(self.last_label_index)

                block = data[self.last_label_index - prestimul_length:self.last_label_index + clapan_length]

                if self.train_flag:
                    self.resultTrain.emit(self.counter, math.log2(label) if label != 0 else 0)
                    self.counter += 1
                    if self.counter == 195:
                        self.stop()
                        self.sendMessage.emit("Требуется заменить животное")
                    if use_auto_train and self.counter >= count_train_stimuls and self.counter % train_step == 0:
                        self.runThreadValidationTrain(data[self.label_index_list[-count_train_stimuls] - prestimul_length:])
                else:
                    if self.is_test_started:
                        self.is_test_started = not self.is_test_started
                        self.counter = 0
                    self.resultTest.emit(self.name, self.counter, self.predict(block),
                                         self.classifierWrapper.convert_result(label))
                    self.counter += 1
                    if self.counter == 50:
                        self.stop()

                # говнокод не трогать
                if data[self.last_corr_index:].shape[0] >= corr_len:
                    self.last_corr_index += corr_len
                    batch = data[self.last_corr_index - corr_len:self.last_corr_index - 1]
                    self.runThreadProcessing(batch)


