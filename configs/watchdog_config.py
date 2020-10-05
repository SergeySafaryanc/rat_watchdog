# Socket conf
HOST = '10.132.230.4'
PORT = 5000

# Worker conf В директорию out_path выводится файл: dat inf с названием train или test.py + время начала записи файлы
# с результатами имеют такое же название как и у dat файла с префиксами: result, _validation_result,
# _responses_classifiers
wait_time = 1  # sec # ожидание детекта
epoch_time = 0.04  # sec # время перерыва чтения из файла
inp_path = "/home/quantum/PycharmProjects/rat_watchdog/data/inp"
out_path = "/home/quantum/PycharmProjects/rat_watchdog/data/out"

sampling_rate = 1000  # частота дискретизации
decimate_rate = 10  # прорежение данных
num_channels = 18  # число каналов ЭГ+дыхание+метка

num_counter_for_refresh_animal = 120  # количество

count_train_stimuls = 60 # число тренировок
train_step = 15  # число смещений (кол-во клапанов * 5)

data_source_is_file = True  # выбор датасорса, true - file, false - socket
is_result_validation = True  # не трогать (типа валидация, кол-во правильных из скольких)

is_train = True  # обучение или тестирование
use_auto_train = True  # обучение сразу после 100-ой подачи (не трограть)

# Доп. запись
# odors = [("Мухамад 1", "#ffff00"), ("Гребеньков 1", "#ffff00"), ("Лежнев 1", "#ffff00")]
odors = [("Доп. запись", "#ffff00"), ("Доп. запись", "#ffff00"), ("Доп. запись", "#ffff00")]
         #("Вертилов 2", "#ffff00"),
         #("Губа 2", "#ffff00"), ("Чуб 2", "#ffff00"),
         # ("5", "#ffff00"), ("3", "#ffff00")
         # ]  # Расшифровка и цвет вещества (на экран и в таблицу)тки веществ (числовые)
#odors_set = [1, 2, 4]  # метки веществ (числовые)
odors_set = [1,2,4]  # метки веществ (числовые)

unite = [[0], [1], [2]]  # передаётся в классификатор для обучения (теста после mixture_groups)

unite_test = [[0], [1], [2]]  # Для подсчета результатов при смене клапанов

# для конвертации приходящих меток
# labels_map = {1: 1, 2: 2, 4: 4} # for test
# labels_map = {8: 2, 16: 3, 32: 5}
labels_map = {1: 0, 2: 1, 4: 2}

result_messages = [("Ожидание", "#CCC")]  # сообщения, выводимые в UI
show_result_delay = 2  # 20 # sec # сколько будет висеть метка

channel_pairs_to_corr = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8], [0, 9], [0, 10], [0, 11],
                         [0, 12]]  # корреляция между каналами (усреднённая, не трогать)
channel_pairs_to_corr_means = True  # не используется

corr_len = 30000  # число отчётов, через которое и на основании которых обновится график

pylib_path = r'../odor_classifier'  # путь к классификатору

clapan_length = 5000  # длина стимула
prestimul_length = 3000  # длина предстимульного отрезка в файле
stimul_delay = 0  # за держка на стимуле. Если пердстимульный отрезок и стимул одинаковые, то =0 else
# stimul_delay=clapan_length-prestimul_length
num_of_channels = num_channels - 2  # количество каналов ЭОГ (всегда на 2 меньше)
# air_clapan = False  # наличие в файлет на последнем канале метки открытия клапана воздуха
# using_air_clapan = False  # использовать клапан воздуха для классификации как стимул
# mixture_groups = [[1], [8], [9], [16]]  # для классификации по клапанам (для Колиного)
mixture_groups = [[1], [2], [4]]  # для классификации по клапанам (для Колиного)
# mixture_groups = [[1, 4], [16, 8, 1]]          #для группировки. Пусть [4,5- Иное], [1,3- НС], [2,6 - ВВ]. Для Иное
# метка-'0', для НС метка-'1', для ВВ метка-'2'
validation_thresh = 80

# костыль, если один файл для обучения и для тестирования, то True
one_file = True
