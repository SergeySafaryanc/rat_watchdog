# from configs.watchdog_config import *
from configs.watchdog_config import *
import numpy as np
import os
import datetime
import math
import scipy.signal
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from itertools import combinations
from tqdm import tqdm
import configparser
import scipy as sp
from scipy.signal import lfilter
from scipy.fftpack import fft
from scipy import stats
from scipy.stats import trim_mean
import sys
import logging
import pickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler,PowerTransformer,Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
logging.getLogger("numpy").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)

np.random.seed(42)

def get_time_stamp(verbose=0):
    time=datetime.datetime.now().timetuple()
    if verbose!=0:
        print('>>>>> ',time[0],'-',time[1],'-',time[2],'  ',time[3],':',time[4],':',time[5])
    time_stamp_list=[time[0],time[1],time[2],time[3],time[4],time[5]]
    time_stamp_string=''.join('_'+str(ind) for ind in time_stamp_list)
    return time_stamp_string

def read_dat_func(file_name_dat, Nch):
    row_data= np.fromfile(file_name_dat, dtype=np.int16)
    l=int(row_data.size/Nch)
    data=row_data.reshape(l, Nch)
    return data

def read_inf_func (file_name_inf):
    config_inf = configparser.ConfigParser()
    config_inf.read(file_name_inf)
    config_inf.sections()
    NChannels_str=config_inf.get("Parameters", "NChannels")
    NChannels=int(NChannels_str)
    NSamplings_str=config_inf.get("Parameters", "NSamplings")
    NSamplings=int(NSamplings_str)
    SamplingFrequency_str=config_inf.get("Parameters", "SamplingFrequency")
    SamplingFrequency=int(SamplingFrequency_str)
    return NChannels,NSamplings,SamplingFrequency

def read_dat(file_name):
    file_name_inf=(file_name[0:-3])+'inf'
    check_exist_inf=os.path.isfile(file_name_inf)   #проверяем ссуществует ли инф файл к нашему дат
    if check_exist_inf==True:
        [NChannels,NSamplings,SamplingFrequency] = read_inf_func(file_name_inf)
        data_matrx=read_dat_func(file_name,NChannels)
    else:
        data_matrx=read_dat_func(file_name,10)
        SamplingFrequency=1000
    return data_matrx,SamplingFrequency

class Clapan:
    """
        Cоздаем класс - Клапан. Каждый объект класса имеет:
        Clapan.openning_ind - индексы открытия клапана
        Сlapan.closing_ind-индексы закрытия клапана
        Сlapan.clap_real_number-реальный номер клапана(на основе преобразования метки)
    """
    def __init__(self, clapan_label, clapan_index,clapan_length,prestimul_length,stimul_delay):
        self.clapan_label = clapan_label
        self.clap_real_number = math.log(clapan_label,2)+1
        bin_clapan_label = np.asarray([int(x) for x in bin(int(self.clapan_label))[2:]])[::-1]
        self.clap_real_number = np.where(bin_clapan_label>0)[0]+1
        self.openning_ind = clapan_index
        self.prestimul_opening = self.openning_ind-prestimul_length
        self.prestimul_closing = self.prestimul_opening+prestimul_length
        self.closing_ind = self.openning_ind+clapan_length
        self.clapans_len = clapan_length
        if stimul_delay!=0:
            self.openning_ind= self.openning_ind+stimul_delay
        pass

def read_dat_by_clapans(data, clapan_length=5000,prestimul_length=0,stimul_delay=0):
    """
        [data, clapans_obj_list] = read_dat_by_clapansfile_name, lst_using_clapans=[],clapan_length=5000,prestimul_length=5000,stimul_delay=0,air_clapan=False,using_air_clapan=False
        Функция считывает dat файл с использовнием объекта __Clapan__
        Позволяет считать DAT-файл с использованием только необходимого списка клапанов
        lst_using_clapans - список используемых клапанов. По-умолчанию = [] ====>> Соответствует использованию всех клапанов кроме последнего(воздуха)
        clapan_length=5000 - длина стимула
        prestimul_length=5000 - длина предстимульного отрезка
        stimul_delay=N - задержка на стимуле. В случае если !=0, то для последующих рассчетов будет браться не вся длина стимула, а длина начиная с N
        air_clapan=False - наличие в файлах метки открытия клапана вооздху(последниго клапана)
        using_air_clapan=False - использовать ли в качестве одного из классов последник клапан(воздух)
    """
    labels_ch = data[:,-1]
    unique_clapans = np.unique(labels_ch)
    unique_clapans = np.delete(unique_clapans, np.where(unique_clapans == 0))
    # if air_clapan==True:
    #     if using_air_clapan==True:
    #         unique_clapans = np.unique(labels_ch)[1:]
    #     else:
    #         np.unique(labels_ch)[1:-1]
    # else:
    #     unique_clapans = np.unique(labels_ch)[1:]
    # if lst_using_clapans!=[-1]:
    #     unique_clapans=np.asarray(lst_using_clapans)
    # unique_clapans = np.asarray([2**(uniq-1) for uniq in unique_clapans])
    mask_by_labels = np.isin(labels_ch,unique_clapans)
    clapans_indexes= np.where(mask_by_labels==True)[0]
    clapans_values = labels_ch[mask_by_labels]
    clapans_obj_list = [Clapan(clapans_values[i], clapans_indexes[i], clapan_length,prestimul_length,stimul_delay) for i in range(clapans_values.shape[0])]
    return data, clapans_obj_list




def butterworth_bandstop_filter(data, lowcut, highcut, Fd=100, order=3):
    nyq = 0.5 * Fd
    low = lowcut / nyq
    high = highcut / nyq
    i, u = sp.signal.butter(order, [low, high], btype='bandstop')
    y = lfilter(i, u, data)
    return y

def butterworth_banpass_filter(data, lowcut, highcut, Fd=100, order=2):
    nyq = 0.5 * Fd
    low = lowcut / nyq
    high = highcut / nyq
    i, u = scipy.signal.butter(order, [low, high], btype='bandpass')
    y = lfilter(i, u, data)
    return y

def AC_filter(input_channel,Fd=1000):
    """
        Return original signal without Alterrnative current noize
        Based on cascade of 3 butterworth_bandstop_filters
        49-51
        99-101
        149-151
    """
    input_channel=butterworth_bandstop_filter(input_channel, lowcut=49, highcut=51, Fd=Fd, order=3)
    input_channel=butterworth_bandstop_filter(input_channel, lowcut=99, highcut=101, Fd=Fd, order=3)
    input_channel=butterworth_bandstop_filter(input_channel, lowcut=149, highcut=151, Fd=Fd, order=3)
    return input_channel

def multichannel_bandpass_filter(input_data,Fd=1000,lowcut=60, highcut=120, order=2):
    for i in range(input_data.shape[1]):
        input_data[:,i]=butterworth_banpass_filter(data=input_data[:,i], lowcut=lowcut, highcut=highcut, Fd=Fd, order=order)
    return input_data

def multichannel_AC_filter(input_data,Fd=1000):
    for i in range(input_data.shape[1]):
        input_data[:,i]=AC_filter(input_channel=input_data[:,i],Fd=Fd)
    return input_data

def get_clapan_data(input_clapan_obj,input_data):
    full_data = input_data[input_clapan_obj.prestimul_opening:input_clapan_obj.closing_ind]
    prestimul_data = input_data[input_clapan_obj.prestimul_opening:input_clapan_obj.prestimul_closing]
    stimul_data = input_data[input_clapan_obj.openning_ind:input_clapan_obj.closing_ind]
    label = input_clapan_obj.clap_real_number
    return prestimul_data, stimul_data, label

###########################################################################


def slope_features_2d(stimul_data):
    channels_list = np.arange(0,stimul_data.shape[1])
    non_overlaping_pairs =np.asarray(list(combinations(channels_list, 2)))
    feature_row = []
    for pair in non_overlaping_pairs:
        sig1 =  stimul_data[:,pair[0]]
        sig2 =  stimul_data[:,pair[1]]
        slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(sig1,sig2)
        feature_row.append(np.asarray([slope1, intercept1, r_value1,  std_err1]))
    feature_row = np.concatenate(feature_row)
    return feature_row





def difference_2d_stimul_features(input_data,input_clapan, feature_type='Slope_original_signal', normirovka = True,Fd=1000):
    """
        input_data - матрица содержащая весь исходный файл
        input_clapan - Объект 'Clapan_obj' для обрабатываемого стимула
        Fd=1000 -частота дискретизации в Гц
        feature_type = - Cor_original_signal - корреляциии между каналами для исходного сигнала
                       - Cor_AC_filtered_signal - корреляции между каналами для фильрованного от наводок сети(50,100, 150 Гц) сигнала
                       - Cor_high_freaq_sinal - корреляции между каналми для сигнала пропущенного через полосовой фильтр в гамма области(60-120 гц)
        normirovka = True - нормировка данных. Если True, то для всех значний корреляций предстимульного отрезка выполняется нормировка путем деления на максимум. 
                            Для отрезка стимула выполняется то же самое. Затем данные объединяются в один массив
    """
    prestimul_data, stimul_data, label = get_clapan_data(input_clapan_obj=input_clapan,input_data=input_data)
    if feature_type=='Slope_original_signal':
        pass
    elif feature_type=='Slope_high_freaq_sinal':
        stimul_data = multichannel_bandpass_filter(stimul_data,Fd,lowcut=60, highcut=120, order=2)
    elif feature_type=='Slope_AC_filtered_signal':
        stimul_data = multichannel_AC_filter(stimul_data,Fd)
    stimul_features = slope_features_2d(stimul_data)
    if normirovka==True:
        stimul_features = stimul_features/max(stimul_features)
    clapan_features = stimul_features
    clapan_label=input_clapan.clapan_label
    clapan_sample = np.append(clapan_features,clapan_label)
    # print(clapan_sample)
    return clapan_sample


def tst_Spl_original_signal(data,SamplingFrequency):

    test_data, test_clapans = read_dat_by_clapans(data, clapan_length=clapan_length,prestimul_length=prestimul_length,stimul_delay=stimul_delay)
    test_dataset = []
    for clapan in test_clapans:
        features =difference_2d_stimul_features(input_data=test_data[:,:-2],input_clapan=clapan, feature_type='Slope_original_signal', normirovka = False,Fd=SamplingFrequency)
        test_dataset.append(features)
    test_dataset=np.vstack(test_dataset)
    test_features = test_dataset[:,:-1]
    test_labels = test_dataset[:,-1]
    model_filename = 'pretrained_model_LDA-ORS.mdl'
    loaded_model = pickle.load(open(model_filename, 'rb'))
    result_loaded = loaded_model.predict(test_features)
    if test_labels.shape[0]>1:
        output = result_loaded
    else:
        output = result_loaded[0]
    print(f"Labels: {test_labels}\tOutput: {output}")
    return output-1
