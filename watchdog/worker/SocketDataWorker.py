import math
import os

import numpy as np

from configs.watchdog_config import *
import socket

from watchdog.worker.AbstractDataWorker import AbstractDataWorker


class SocketDataWorker(AbstractDataWorker):

    def __init__(self, bytes_to_read, decimate_rate, socket_num_channels, channel_pairs, is_train):

        super().__init__(bytes_to_read, decimate_rate, channel_pairs, ("train" if is_train else "test"), is_train)
        self.socket_num_channels = socket_num_channels
        self.label_index_list = []
        self.working = False

    def run(self):
        self.working = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            odor_labels_set = set(odors_set)
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            size_read = 0
            self.record = np.array([]).reshape(-1, self.socket_num_channels)
            data = np.array([]).reshape(-1, num_channels)

            with conn:
                print('Connected by', addr)
                while self.working:
                    buffer = np.frombuffer(conn.recv(self.bytes_to_read, socket.MSG_WAITALL), 'i2').reshape(
                        (-1, self.socket_num_channels))
                    self.record = np.vstack((self.record, np.copy(buffer)))

                    data = np.vstack(
                        (data, self.decimate(np.copy(buffer)[:, :num_channels], k=self.decimate_rate, method='')))

                    for i in range(size_read, data.shape[0]):
                        if {data[i, -1]}.issubset(odor_labels_set.difference({self.current_label})):
                            self.current_label = data[i, -1]
                            self.last_label_index = i
                            label = data[i, -1]
                        else:
                            if data[i, -1] == 64:
                                self.current_label = -1
                            data[i, -1] = 0

                    with open(os.path.join(out_path, self.path_to_res + ".dat"), 'ab') as f:
                        np.copy(data[size_read:]).reshape(-1).astype('int16').tofile(f)
                    size_read = data.shape[0]

                    self.create_inf(self.path_to_res, size_read)

                    # for test.py
                    # self.tickViewSig.emit(data[:, num_channels - 1])

                    if (data[self.last_label_index:].shape[0] < clapan_length) or (
                            data[:self.last_label_index].shape[0] < prestimul_length):
                        continue

                    block = np.copy(
                        data[self.last_label_index - prestimul_length:self.last_label_index + clapan_length])

                    self.label_index_list.append(self.last_label_index)

                    if is_train:
                        self.resultTrain.emit(self.counter, labels_map[label])
                        self.counter += 1
                        if self.counter == 195:
                            self.stop()
                            self.sendMessage.emit("Требуется заменить животное")
                        if use_auto_train and self.counter >= count_train_stimuls and self.counter % train_step == 0:
                            self.runThreadValidationTrain(data[self.label_index_list[-count_train_stimuls] - prestimul_length:])
                    else:
                        self.resultTest.emit(self.time_now, self.counter, self.predict(block),
                                             self.classifierWrapper.convert_result(labels_map[label]))
                        self.counter += 1


                    self.last_label_index = 0
                    # говнокод не трогать
                    if data[self.last_corr_index:].shape[0] >= corr_len:
                        self.last_corr_index += corr_len
                        batch = np.copy(data[self.last_corr_index - corr_len:self.last_corr_index - 1])
                        self.runThreadProcessing(batch)

    def decimate(self, data, k=10, method="mean"):
        if method == "mean":
            x = np.reshape(data[:k * (len(data) // k)], (-1, k, data.shape[-1]))
            x = x.mean(axis=1)
        else:
            x = data[::k]
        return x
