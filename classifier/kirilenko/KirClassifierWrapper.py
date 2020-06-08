from classifier.kirilenko import kir_train_seq_SLP1, kir_train_seq_SLP_LDA, kir_train_seq_SLP_LDA_PT, kir_test_seq_SLP1, \
    kir_test_seq_SLP_LDA, kir_test_seq_SLP_LDA_PT
import numpy as np

class KirClassifierWrapper:
    def __init__(self):
        pass

    def train(self, train_file_path):
        res = []
        res.append(kir_train_seq_SLP1.train(train_file_path))
        # res.append(kir_train_seq_SLP_LDA.train(train_file_path))
        res.append(kir_train_seq_SLP_LDA_PT.train(train_file_path))
        return res


    def predict(self, data):
        res = []
        res.append(int(kir_test_seq_SLP1.tst_Spl_original_signal(data)))
        # res.append(int(kir_test_seq_SLP_LDA.tst_Spl_original_signal(data)))
        res.append(int(kir_test_seq_SLP_LDA_PT.tst_Spl_original_signal(data)))
        return res
