import math
from sys import path as pylib
import os
from configs.watchdog_config import pylib_path, unite, unite_test

pylib += [os.path.abspath(pylib_path)]
from rat import *
from loguru import logger

class ClassifierWrapper:
    def __init__(self, num_channels=16, odors=[1, 2, 4, 8, 16], shep_groups=[[1, 2], [4, 8], [16]], decimate=1):
        """

        @param num_channels:
        @param odors:
        @param unite:
        @param decimate:
        @param unite_test: Нужно для подсчета результатов, когда меняем местами клапана
        """
        self.num_channels = num_channels
        self.odors = odors
        self.shep_groups = shep_groups
        self.decimate = decimate
        self.unite = unite
        self.unite_test = unite_test

        logger.info(f"ClassifierWrapper.py: num_channels: {self.num_channels}")
        logger.info(f"ClassifierWrapper.py: odors: {self.odors}")
        logger.info(f"ClassifierWrapper.py: shep_group: {self.shep_groups}")
        logger.info(f"ClassifierWrapper.py: decimate: {self.decimate}")
        logger.info(f"ClassifierWrapper.py: unite: {self.unite}")
        logger.info(f"ClassifierWrapper.py: unite_test: {self.unite_test}")

    def predict(self, data):
        # unite_index = [i[0] for i in self.unite]
        logger.info("<<<<<<<<<<<<<<<<<")
        logger.info(self.odors)
        logger.info(classifier(data, self.num_channels, self.odors, [], self.decimate))
        logger.info(">>>>>>>>>>>>>>>>")
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
            for j in range(len(self.unite)):
                if res[i] in self.unite[j]:
                    result.append(j)
        logger.info(f"ClassifierWrapper.py: result: {result}")
        return result