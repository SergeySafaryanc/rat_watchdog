from configs.watchdog_config import *
from watchdog.utils.readme import Singleton, write
from watchdog.worker.AbstractDataWorker import AbstractDataWorker
from time import sleep
import numpy as np

from loguru import logger
from itertools import chain
import os

class FileDataWorker(AbstractDataWorker):
    def __init__(self, f_name, name, bytes_to_read, epoch_time, decimate_rate, channel_pairs, train_flag):
        super().__init__(bytes_to_read, decimate_rate, channel_pairs, name, train_flag)
        self.working = True
        self.f_name = f_name
        self.name = name
        self.is_test_started = True
        self.epoch_time = epoch_time
        self.label_index_list = []
        logger.info(f"f_name = {self.f_name}\tname = {self.name}\tepoch_time = {self.epoch_time}")

    def run(self):
        find_label = True
        with open(self.f_name, 'rb', buffering=0) as f:
            odor_labels_set = set(list(chain(*odors_unite)))
            size_read = 0
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

                for i in range(size_read, data.shape[0]):
                    if {data[i, -1]}.issubset(odor_labels_set.difference({self.current_label})):
                        if i - self.last_label_index < 500:  # проверка на двойную метку
                            logger.info(i)
                            logger.info(i - self.last_label_index)
                            data[i, -1] = 0
                            continue
                        self.current_label = data[i, -1]
                        self.last_label_index = i
                        label = data[i, -1]
                        # logger.info(self.current_label)
                    else:
                        if data[i, -1] == 0:
                            self.current_label = -1
                        data[i, -1] = 0
                    # logger.info(self.current_label)

                if self.train_flag and (len(odors_unite) != len(list(chain(*odors_unite)))):
                    # здесь скорректировать
                    data = self.correct_labels_by_groups(data)
                    # сохранить как .dat
                    with open(os.path.join(self.exp_folder, self.path_to_res + ".dat"), 'ab') as d:
                        # with open(os.path.join(out_path, self.path_to_res + ".dat"), 'ab') as d:
                        np.copy(data[size_read:]).reshape(-1).astype('int16').tofile(d)

                    size_read = data.shape[0]

                    self.create_inf(self.path_to_res, size_read)

                size_read = data.shape[0]

                if (data[self.last_label_index:].shape[0] < clapan_length) or (
                        data[:self.last_label_index].shape[0] < prestimul_length):
                    continue

                block = np.copy(
                    data[self.last_label_index - prestimul_length:self.last_label_index + clapan_length])

                self.label_index_list.append(self.last_label_index)

                if self.train_flag:
                    self.resultTrain.emit(self.counter, self.labels_map[label])
                    self.counter += 1
                    if self.counter == num_counter_for_refresh_animal:
                        self.stop()
                        Singleton.set("Результат", "Требуется заменить животное")
                        write(Singleton.text())
                        self.sendMessage.emit("Требуется заменить животное")
                        #TODO fix counter %
                    if use_auto_train and self.counter >= count_train_stimuls and self.counter % train_step == 0:
                        self.runThreadValidationTrain(data[self.label_index_list[-count_train_stimuls] - prestimul_length:])
                else:
                    if self.is_test_started:
                        self.is_test_started = not self.is_test_started
                        self.counter = 0
                    self.resultTest.emit(self.name, self.counter, self.predict(block),
                                         self.classifierWrapper.convert_result(self.labels_map[label]))
                    self.counter += 1
                    if self.counter == 75:  # количество подач на тест
                        self.stop()

                self.last_label_index = 0
                # говнокод не трогать
                if data[self.last_corr_index:].shape[0] >= corr_len:
                    self.last_corr_index += corr_len
                    batch = data[self.last_corr_index - corr_len:self.last_corr_index - 1]
                    self.runThreadProcessing(batch)


