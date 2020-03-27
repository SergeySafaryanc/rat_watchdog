from classifier.kirilenko.kir_test_seq_SLP1 import pearson_corr_for_2d_signal
from configs.watchdog_config import *
import numpy as np
import os
import datetime
import math
import scipy.signal
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
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
        self.openning_ind = clapan_index
        self.prestimul_opening = self.openning_ind-prestimul_length
        self.prestimul_closing = self.prestimul_opening+prestimul_length
        self.closing_ind = self.openning_ind+clapan_length
        self.clapans_len = clapan_length
        if stimul_delay!=0:
            self.openning_ind= self.openning_ind+stimul_delay
        pass

def read_dat_by_clapans(file_name, lst_using_clapans=[-1],clapan_length=5000,prestimul_length=5000,stimul_delay=0,air_clapan=False,using_air_clapan=False):
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
    data,SamplingFrequency = read_dat(file_name)
    labels_ch = data[:,data.shape[1]-1]
    if air_clapan==True:
        if using_air_clapan==True:
            unique_clapans = np.unique(labels_ch)[1:]
        else:
            np.unique(labels_ch)[1:-1]
    else:
        unique_clapans = np.unique(labels_ch)[1:]
    if lst_using_clapans!=[-1]:
        unique_clapans=np.asarray(lst_using_clapans)
    unique_clapans = np.asarray([2**(uniq-1) for uniq in unique_clapans])
    mask_by_labels = np.isin(labels_ch,unique_clapans)
    clapans_indexes= np.where(mask_by_labels==True)[0]
    clapans_values = labels_ch[mask_by_labels]
    clapans_obj_list = [Clapan(clapans_values[i], clapans_indexes[i], clapan_length,prestimul_length,stimul_delay) for i in range(clapans_values.shape[0])]
    return data, clapans_obj_list, SamplingFrequency

class Classifier_obj:
    """
        Classifier
        model_types:
            -RandomForest
            -DecisionTrees
            -LinearSVM
        METHODS:
        create_object-->   clf = Classifier_obj(model_type='RandomForest')
        fit-->   clf.fit(train_features,train_features)
        predict-->   result = clf.predict(input_features) -get predict list
        save-->   save_model(path_to_model)
        load_model-->   load_model(self,path_to_model)
    """
    def __init__(self, model_type='RandomForest'):
        """
            clf = Classifier_obj(model_type='RandomForest')
            model_types:
          -RandomForest
          -DecisionTrees
          -LinearSVM
        """
        if model_type=='DTrees' or model_type=='Dtrees' or model_type=='DTree' or model_type=='Dtree' or model_type=='DecisionTrees':
            self.model = tree.DecisionTreeClassifier()
            self.model_name = 'DTrees'
        elif model_type=='LinearSVM' or model_type=='LSVM' or model_type=='SVM':
            self.model = SVC(gamma='auto')
            self.model_name ='LSVM'
        elif model_type=='RForest' or model_type=='Rforest' or model_type=='RFores' or model_type=='RandomForest':
            self.model = RandomForestClassifier(n_estimators=150, criterion='entropy', max_depth=None,
                                                min_samples_split=4, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                                max_features='auto', max_leaf_nodes=None,
                                                bootstrap=True, oob_score=False, n_jobs=-1,
                                                random_state=None, verbose=0, warm_start=False,
                                                class_weight=None)
            self.model_name ='RForest'
        elif model_type=="GBoost" or model_type=="GBoosting":
            self.model = GradientBoostingClassifier(loss='deviance', learning_rate=0.012,n_estimators=50,subsample=0.2,criterion='friedman_mse',min_samples_split=3,max_depth=3,random_state=42)
        else:
            print('Undefined model!!!!')
        pass
    def fit(self, input_features, input_labels):
        """
            clf.train(train_features,train_features)
            train_featurs - 2d numpy array, where cols=features, rows=samples
            train_labels - 1d numpy array with strings or digits
            This mehthod train or retrain model with input data
        """
        unique_labels = np.unique(input_labels)
        self.num_of_features = input_features.shape[1]
        keys_of_classes =[i for i in range(len(unique_labels))]
        self.labels_dictionary = dict(zip(keys_of_classes, unique_labels))
        self.dictionary_table = np.asarray([[key,value] for key,value in self.labels_dictionary.items()])
        train_labels = np.asarray([np.where(self.dictionary_table[:,1]==i)[0][0] for i in input_labels], dtype=np.int32)
        self.model.fit(input_features,train_labels)
    def refit(self, input_features, input_labels):
        """
            clf.train(train_features,train_features)
            train_featurs - 2d numpy array, where cols=features, rows=samples
            train_labels - 1d numpy array with strings or digits
            This mehthod REtrain model with input data
        """
        retrain_labels =np.asarray([np.where(self.dictionary_table[:,1]==i)[0][0] for i in input_labels])
        self.model.fit(input_features,retrain_labels)
        pass
    def save_model(self,path_to_model):
        pickle.dump(self.model, open(path_to_model, 'wb'))
        pass
    def load_model(self,path_to_model):
        self.model = pickle.load(open(path_to_model, 'rb'))
        pass
    def predict(self,test_features):
        result = self.model.predict(test_features)
        return np.asarray([self.labels_dictionary[i] for i in result])
    def predict_classes(self,test_features):
        result = self.model.predict(test_features)
        return result+1
###########################################################################
###########################################################################



def get_fft_subset(input_channel,Fd,low_freq,high_freq, mode='PowerSpectrum'):
    """ Fd - sampling frequency
        input_data - 1d array
        mode =
                PowerSpectrum
                Magnitude"""
    df = Fd/input_channel.shape[0]
    freq_axis = np.arange(0,input_channel.shape[0]/2,df)
    Zxx = fft(input_channel)
    if mode=='Magnitude':
        Zxx = np.abs(Zxx)
    elif mode=='PowerSpectrum':
        Zxx = np.abs(Zxx)**2
    Zxx = Zxx[:int(Zxx.shape[0]/2)]
    choosed_coef = np.where((freq_axis<=high_freq)&(freq_axis>=low_freq))
    choosed_freq = freq_axis[choosed_coef]
    choosed_Zxx = Zxx[choosed_coef]
    return choosed_Zxx,choosed_freq

def get_fft_multiple_subset_sum(input_channel,Fd, frequency_subbands=[[8,13],[14,30],[31,49],[51,70],[71,99],[101,130],[130,149],[60,120]], mode='PowerSpectrum',aver_mode= 'trim_mean'):
    """ Fd - sampling frequency
        input_data - 1d array
        mode =
                PowerSpectrum
                Magnitude
            frequency_subbands = [[1,3],
                              [4,7],
                              [8,13],
                              [14,30],
                              [31,49],
                              [51,70],
                              [71,99],
                              [101,130],
                              [130,149],
                              [60,120]]
         aver_mode ==    'trim_mean'
                        'mean'
                        'sum'
                """
    df = Fd/input_channel.shape[0]
    freq_axis = np.arange(0,input_channel.shape[0]/2,df)
    Zxx = fft(input_channel)
    if mode=='Magnitude':
        Zxx = np.abs(Zxx)
    elif mode=='PowerSpectrum':
        Zxx = np.abs(Zxx)**2
    Zxx = Zxx[:int(Zxx.shape[0]/2)]
    freq_axis = freq_axis[:Zxx.shape[0]]
    averaged_on_subband_values = []
    for j in range(len(frequency_subbands)):
        low_freq = frequency_subbands[j][0]
        high_freq = frequency_subbands[j][1]
        choosed_coef = []
        for i in range(freq_axis.shape[0]):
            if (freq_axis[i]<=high_freq and freq_axis[i]>=low_freq):
                choosed_coef.append(i)
        choosed_Zxx = Zxx[choosed_coef]
        averaged_on_subband = 0
        if aver_mode == 'trim_mean':
            averaged_on_subband = trim_mean(choosed_Zxx, 0.1)
        elif aver_mode == 'mean':
            averaged_on_subband = np.mean(choosed_Zxx)
        elif aver_mode == 'sum':
            averaged_on_subband = np.sum(choosed_Zxx)
        averaged_on_subband_values.append(averaged_on_subband)
    return np.asarray(averaged_on_subband_values)


def get_multichannel_fft_features(input_data,Fd, frequency_subbands=[[31,49],[51,70],[71,99],[101,130],[50,130],[60,120]], mode='PowerSpectrum',aver_mode= 'trim_mean'):
    multichannel_fft_features = np.concatenate([get_fft_multiple_subset_sum(input_channel=input_data[:,ch],Fd=Fd, frequency_subbands=frequency_subbands, mode=mode,aver_mode=aver_mode) for ch in range(input_data.shape[1])])
    return multichannel_fft_features

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
    full_data = input_data[input_clapan_obj.prestimul_opening:input_clapan_obj.closing_ind,:-1]
    prestimul_data = input_data[input_clapan_obj.prestimul_opening:input_clapan_obj.prestimul_closing,:-1]
    stimul_data = input_data[input_clapan_obj.openning_ind:input_clapan_obj.closing_ind,:-1]
    label = input_clapan_obj.clap_real_number
    return prestimul_data, stimul_data, label

###########################################################################

def correlation_stimul_features(input_data,input_clapan, feature_type='Cor_original_signal', normirovka = True,Fd=1000):
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
    if feature_type=='Cor_original_signal':
        pass
    elif feature_type=='Cor_high_freaq_sinal':
        prestimul_data = multichannel_bandpass_filter(prestimul_data,Fd,lowcut=60, highcut=120, order=2)
        stimul_data = multichannel_bandpass_filter(stimul_data,Fd,lowcut=60, highcut=120, order=2)
    elif feature_type=='Cor_AC_filtered_signal':
        prestimul_data = multichannel_AC_filter(prestimul_data,Fd)
        stimul_data = multichannel_AC_filter(stimul_data,Fd)
    prestimul_features = pearson_corr_for_2d_signal(prestimul_data,num_of_channels)
    stimul_features = pearson_corr_for_2d_signal(stimul_data,num_of_channels)
    if normirovka==True:
        prestimul_features = prestimul_features/max(prestimul_features)
        stimul_features = stimul_features/max(stimul_features)
    clapan_features = np.concatenate((prestimul_features,stimul_features))
    clapan_label=input_clapan.clap_real_number
    clapan_sample = np.append(clapan_features,clapan_label)
    return clapan_sample

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



def power_spectrum_stimul_features(input_data,input_clapan, feature_type='PS_original_signal', normirovka = True,Fd=1000):
    """
        input_data - матрица содержащая весь исходный файл
        input_clapan - Объект 'Clapan_obj' для обрабатываемого стимула
        Fd=1000 -частота дискретизации в Гц
        feature_type = - PS_original_signal - спектр мощностии на каждом канале для исходного сигнала
                       - PS_AC_filtered_signal - спектр мощностии на каждом канале для фильрованного от наводок сети(50,100, 150 Гц) сигнала
                       - PS_high_freaq_sinal - спектр мощностии на каждом канале для сигнала пропущенного через полосовой фильтр в гамма области(60-120 гц)
        normirovka = True - нормировка данных. Если True, то для всех значний спектра мощностии на каждом канале предстимульного отрезка выполняется нормировка путем деления на максимум. 
                            Для отрезка стимула выполняется то же самое. Затем данные объединяются в один массив
    """
    prestimul_data, stimul_data, label = get_clapan_data(input_clapan_obj=input_clapan,input_data=input_data)
    if feature_type=='PS_original_signal':
        pass
    elif feature_type=='PS_high_freaq_sinal':
        prestimul_data = multichannel_bandpass_filter(prestimul_data,Fd,lowcut=60, highcut=120, order=2)
        stimul_data = multichannel_bandpass_filter(stimul_data,Fd,lowcut=60, highcut=120, order=2)
    elif feature_type=='PS_AC_filtered_signal':
        prestimul_data = multichannel_AC_filter(prestimul_data,Fd)
        stimul_data = multichannel_AC_filter(stimul_data,Fd)
    prestimul_features = get_multichannel_fft_features(input_data=prestimul_data,Fd=Fd, frequency_subbands=[[31,49],[51,70],[71,99],[101,130],[50,130],[60,120]], mode='PowerSpectrum',aver_mode= 'trim_mean')
    stimul_features = get_multichannel_fft_features(input_data=stimul_data,Fd=Fd, frequency_subbands=[[31,49],[51,70],[71,99],[101,130],[50,130],[60,120]], mode='PowerSpectrum',aver_mode= 'trim_mean')
    if normirovka==True:
        channel_features_pairs = [np.asarray([prestimul_features[feat],stimul_features[feat]]) for feat in range(stimul_features.shape[0])]
        normed_pairs = [pair/max(pair) for pair in channel_features_pairs]
        clapan_features = np.concatenate(normed_pairs)
    else:    
        clapan_features = np.concatenate((prestimul_features,stimul_features))
        clapan_features = clapan_features/max(clapan_features)
    clapan_label=input_clapan.clap_real_number
    clapan_sample = np.append(clapan_features,clapan_label)
    return clapan_sample


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
    stimul_features = slope_features_2d(stimul_data[:,0:num_of_channels])
    if normirovka==True:
        stimul_features = stimul_features/max(stimul_features)
    clapan_features = stimul_features
    clapan_label=input_clapan.clap_real_number
    clapan_sample = np.append(clapan_features,clapan_label)
    return clapan_sample


def slope_original_signal(data):
    train_data, train_clapans, Fd = data
    print('Подготовка признаков')
    train_dataset = []
    for clapan in tqdm(train_clapans):
        features =difference_2d_stimul_features(input_data=train_data[:,:(num_of_channels+1)],input_clapan=clapan, feature_type='Slope_original_signal', normirovka = False,Fd=Fd)
        train_dataset.append(features)
    train_dataset=np.vstack(train_dataset)
    newlabels = []
    for orig_label in train_dataset[:,-1]:
        label_mixt_type = np.where(np.asarray([orig_label in group for group in mixture_groups])==True)[0][0]
        newlabels.append(label_mixt_type)
    newlabels=np.asarray(newlabels)
    train_dataset[:,-1] = newlabels
    train_features = train_dataset[:,:-1]
    train_labels = train_dataset[:,-1]
    clf7 = GradientBoostingClassifier(loss='deviance', learning_rate=0.012,n_estimators=50,subsample=0.2,criterion='friedman_mse',min_samples_split=3,max_depth=3,random_state=42)
    clf7.fit(train_features,train_labels)
    model_filename = 'pretrained_model_SLP-ORS.mdl'
    pickle.dump(clf7, open(model_filename, 'wb'))
    print('Model was trained')

def slope_original_signal_val_last(data):
    train_data, train_clapans, Fd = data
    test_procent = 0.15
    train_dataset = []
    for clapan in tqdm(train_clapans):
        features = difference_2d_stimul_features(input_data=train_data[:, :(num_of_channels + 1)],
                                                 input_clapan=clapan, feature_type='Slope_original_signal',
                                                 normirovka=False, Fd=Fd)
        train_dataset.append(features)
    train_dataset = np.vstack(train_dataset)
    newlabels = []
    for orig_label in train_dataset[:, -1]:
        label_mixt_type = np.where(np.asarray([orig_label in group for group in mixture_groups]) == True)[0][0]
        newlabels.append(label_mixt_type)
    newlabels = np.asarray(newlabels)
    train_dataset[:, -1] = newlabels
    all_features = train_dataset[:, :-1]
    all_labels = train_dataset[:, -1]
    test_num = int(np.round(all_labels.shape[0] * test_procent))
    train_num = all_labels.shape[0] - test_num
    train_features = all_features[:train_num, :]
    train_labels = all_labels[:train_num]
    val_features = all_features[train_num:, :]
    val_labels = all_labels[train_num:]
    clf8 = GradientBoostingClassifier(loss='deviance', learning_rate=0.012, n_estimators=50, subsample=0.2,
                                      criterion='friedman_mse', min_samples_split=3, max_depth=3, random_state=42)
    clf8.fit(train_features, train_labels)
    sc = clf8.predict(val_features)
    return (6, sc)

def train(filename):
    slope_original_signal(read_dat_by_clapans(file_name=filename, lst_using_clapans=sum(mixture_groups, []), clapan_length=clapan_length,
                        prestimul_length=prestimul_length, stimul_delay=stimul_delay, air_clapan=air_clapan,
                        using_air_clapan=using_air_clapan))
    return slope_original_signal_val_last(read_dat_by_clapans(file_name=filename, lst_using_clapans=sum(mixture_groups, []), clapan_length=clapan_length,
                        prestimul_length=prestimul_length, stimul_delay=stimul_delay, air_clapan=air_clapan,
                        using_air_clapan=using_air_clapan))

if __name__ == "__main__":
    train_path = sys.argv[1]
    train(train_path)