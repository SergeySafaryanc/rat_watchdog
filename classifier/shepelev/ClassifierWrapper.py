import math
from sys import path as pylib
import os
from configs.watchdog_config import pylib_path, unite_test

pylib += [os.path.abspath(pylib_path)]
from rat import *


class ClassifierWrapper:
    def __init__(self, num_channels=16, odors=[1, 2, 4, 8, 16], unite=[[1, 2], [4, 8], [16]], decimate=1):
        """

        @param num_channels:
        @param odors:
        @param unite:
        @param decimate:
        @param unite_test: Нужно для подсчета результатов, когда меняем местами клапана
        """
        self.num_channels = num_channels
        self.odors = odors
        self.unite = unite
        self.decimate = decimate
        self.unite_test = unite_test

    def predict(self, data):
        # unite_index = [i[0] for i in self.unite]
        return classifier(data, self.num_channels, self.odors, [], self.decimate)

    def train(self, file_name):
        return train(file_name, self.num_channels, self.odors, [], self.decimate)

    def convert_result(self, label):
        for j in range(len(self.unite_test)):
            if int(label) in self.unite_test[j]:
                return j

    def convert_result_log(self, res):
        result = []
        for i in range(len(res)):
            for j in range(len(unite_test)):
                if res[i] in [math.log(it1, 2) for it1 in unite_test[j]]:
                    result.append(j)
        return result