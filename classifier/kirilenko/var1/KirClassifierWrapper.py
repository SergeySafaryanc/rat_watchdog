from classifier.kirilenko.var1 import kir_train_seq_SLP1, kir_train_seq_SLP_LDA, kir_test_seq_SLP1, kir_test_seq_SLP_LDA
from configs.watchdog_config import sampling_rate


class KirClassifierWrapper1:
    def __init__(self):
        pass

    def train(self, train_file_path):
        res = []
        res.append(kir_train_seq_SLP1.train(train_file_path))
        res.append(kir_train_seq_SLP_LDA.train(train_file_path))
        return res


    def predict(self, data):
        res = []
        res.append(int(kir_test_seq_SLP1.tst_Spl_original_signal(data, sampling_rate)))
        res.append(int(kir_test_seq_SLP_LDA.tst_Spl_original_signal(data, sampling_rate)))
        return res
