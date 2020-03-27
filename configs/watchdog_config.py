#Socket conf
HOST = '10.132.230.4'
PORT = 5000

#Worker conf
# В директорию out_path выводится файл:
    # dat inf с названием train или test.py + время начала записи
    # файлы с результатами имеют такое же название как и у dat файла с префиксами: result, _validation_result, _responses_classifiers
wait_time = 1 # sec
epoch_time = 0.09 # sec
inp_path = "/home/maxburbelov/plexon/input"
out_path = "/home/maxburbelov/plexon/output"



sampling_rate = 1000
decimate_rate = 10
num_channels = 18

count_train_stimuls = 125
train_step = 25

data_source_is_file = True
is_result_validation = True

is_train = False
use_auto_train = True

odors = [("Взрывчатое", "#ffff00"), ("Наркотическое", "#ffff00"), ("Иное", "#ffff00"), ("4", "#ffff00"), ("5", "#ffff00"), ("3", "#ffff00")]

odors_set = [1, 2, 4, 8, 16]

unite = []

unite_test = [[1, 2, 4], [8, 16]]         #Для подсчета результатов при смене клапанов

odors_true = [0, 1, 2, 3, 4]


result_messages = [("Ожидание", "#CCC")]
show_result_delay = 2 #20 # sec

channel_pairs_to_corr = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11], [0, 12]]
channel_pairs_to_corr_means = True

corr_len = 30000

pylib_path = r'../odor_classifier'

clapan_length = 5000                        # длина стимула
prestimul_length = 3000                     #длина предстимульного отрезка в файле
stimul_delay = 0                            # за держка на стимуле. Если пердстимульный отрезок и стимул одинаковые, то =0 else stimul_delay=clapan_length-prestimul_length
num_of_channels = 16                        # количество каналав ЭОГ
air_clapan = False                           #наличие в файлет на последнем канале метки открытия клапана воздуха
using_air_clapan=False                      # использовать клапан воздуха для классификации как стимул
mixture_groups = [[1],[2],[3],[4],[5]]     #для классификации по клапанам
# mixture_groups = [[1, 2, 3], [4, 5]]          #для группировки. Пусть [4,5- Иное], [1,3- НС], [2,6 - ВВ]. Для Иное метка-'0', для НС метка-'1', для ВВ метка-'2'
validation_thresh = 70