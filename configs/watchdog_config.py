# Socket conf
HOST = '10.132.230.4'
PORT = 5000

# Worker conf
# В директорию out_path выводится файл:
# dat inf с названием train или test.py + время начала записи
# файлы с результатами имеют такое же название как и у dat файла с префиксами: result, _validation_result, _responses_classifiers
wait_time = 1  # sec # ожидание детекта
epoch_time = 0.04  # sec # время перерыва чтения из файла
inp_path = "C:\\Users\\JackHuman\\WatchdogFiles\\input"
out_path = "C:\\Users\\JackHuman\\WatchdogFiles\\output"

sampling_rate = 1000  # частота дискретизации
decimate_rate = 10  # прорежение данных
num_channels = 17  # число каналов ЭГ+дыхание+метка

count_train_stimuls = 100  # число тренировок
train_step = 25  # число смещений

data_source_is_file = True  # выбор датасорса, true - file, false - socket
is_result_validation = True  # не трогать (типа валидация, кол-во правильных из скольких)

is_train = True  # обучение или тестирование
use_auto_train = True  # обучение сразу после 100-ой подачи (не трограть)


odors = [("ТНТ", "#ffff00"), ("КП НП", "#ffff00"), ("Воздух из баллона", "#ffff00"), ("ТНТ + КП НП", "#ffff00"),
         ("5", "#ffff00"), ("3", "#ffff00")]  # Расшифровка и цвет вещества (на экран и в таблицу)

odors_set = [1, 8, 16, 32]  # метки веществ (числовые)

unite = [[1], [8], [16], [32]]  # передаётся в классификатор для обучения

unite_test = [[1], [8], [16], [32]]  # для подсчета результатов при смене веществ на клапанах

odors_true = [0, 1, 2, 3, 4]  # не используется

result_messages = [("Ожидание", "#CCC")]  # сообщения, выводимые в UI
show_result_delay = 2  # 20 # sec # сколько будет висеть метка

channel_pairs_to_corr = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11],
                         [0, 12]]  # корреляция между каналами (усреднённая, не трогать)
channel_pairs_to_corr_means = True  # не используется

corr_len = 30000  # число отчётов, через которое и на основании которых обновится график

pylib_path = r'../odor_classifier'  # путь к классификатору

clapan_length = 5000  # длина стимула
prestimul_length = 3000  # длина предстимульного отрезка в файле
stimul_delay = 0  # задержка на стимуле. Если пердстимульный отрезок и стимул одинаковые, то =0 else stimul_delay=clapan_length-prestimul_length
num_of_channels = 15  # количество каналав ЭОГ
air_clapan = False  # наличие в файлет на последнем канале метки открытия клапана воздуха
using_air_clapan = False  # использовать клапан воздуха для классификации как стимул
mixture_groups = [[1], [2], [3], [4], [5]]  # для классификации по клапанам
# mixture_groups = [[1, 2, 3], [4, 5]]          #для группировки. Пусть [4,5- Иное], [1,3- НС], [2,6 - ВВ]. Для Иное метка-'0', для НС метка-'1', для ВВ метка-'2'
validation_thresh = 80
