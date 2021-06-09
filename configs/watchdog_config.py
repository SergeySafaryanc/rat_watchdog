# Socket conf
HOST = 'localhost'
# HOST = '10.132.230.4'
PORT = 4000  # 5000 если передача удалённая, 4000 если с этого же устройства

# Worker conf В директорию out_path выводится файл: dat inf с названием train или test.py + время начала записи файлы
# с результатами имеют такое же название как и у dat файла с префиксами: result, _validation_result,
# _responses_classifiers
wait_time = 1  # sec # ожидание детекта
epoch_time = 0.04  # sec # время перерыва чтения из файла
inp_path = "C:\\WatchdogFiles\\input"  # для Windows
out_path = "C:\\WatchdogFiles\\output"  # для Windows
# inp_path = "/home/quantum/Documents/watchdog_files/inp"  # для Linux
# out_path = "/home/quantum/Documents/watchdog_files/out"  # для Linux

sampling_rate = 1000  # частота дискретизации
decimate_rate = 1  # прорежение данных, 10 для PLEXON онлайн, 1 для ТД1 и оффлайн (.dat)
num_channels = 18  # число каналов ЭГ+дыхание+метка

num_clapans = 4  # число клапанов для автогенерации

num_counter_for_refresh_animal = num_clapans*44  # количество стимулов для смены животного
count_train_stimuls = num_clapans*25  # число тренировок
train_step = num_clapans*5  # число смещений (кол-во клапанов * 5)

data_source_is_file = True  # выбор датасорса, true - file, false - socket
is_result_validation = True  # не трогать (типа валидация, кол-во правильных из скольких)

is_train = False  # обучение или тестирование
use_auto_train = True  # обучение сразу после достижения count_train_stimuls (не трограть)

# odors = [("Мухамад 1", "#ffff00"), ("Гребеньков 1", "#ffff00"), ("Лежнев 1", "#ffff00")]
# odors = [("ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00")]
# odors = [("ТНТ", "#ffff00"), ("Приправа", "#ffff00"), ("Корвалол", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("Героин", "#ffff00"), ("Амфетамин", "#ffff00"), ("ТНТ", "#ffff00"),
#          ("октоген", "#ffff00"), ("тетрил", "#ffff00"), ("Гексоген 10-15", "#ffff00"), ("воздух", "#ffff00")]
# odors = [("ТНТ", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("ВВ", "#ffff00"), ("КП", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("ВВ", "#ffff00"), ("НС", "#ffff00"), ("НС", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("ВВ", "#ffff00"), ("НС", "#ffff00"), ("НС", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00")]
# odors = [("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"),
#          ("ЦВ обнаружено", "#ffff00"), ("ЦВ обнаружено", "#ffff00"), ("ЦВ обнаружено", "#ffff00"),
#          ("Х", "#ffff00"), ("Х", "#ffff00")]
# odors = [("ВВ", "#ffff00"), ("ВВ", "#ffff00"), ("НС", "#ffff00"), ("НС", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00")]

odors = [("ВВ", "#ffff00"), ("ВВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00")]


# odors_set = [1, 2, 4]  # не используется уже! метки веществ с клапанов, передаётся в классификатор для обучения

# odors_unite = [[1], [2], [4], [8], [16], [32]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения
# odors_unite = [[1], [2], [4], [8, 16, 32]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения
# odors_unite = [[1, 2], [4], [8, 16, 32, 64, 128]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения
# odors_unite = [[1, 2, 4], [8, 16], [32]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения
# odors_unite = [[1, 2, 4, 8], [16], [32], [64]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения

odors_unite = [[1], [2], [16], [32]]  # метки веществ клапанов по группам в порядке!, передаётся в классификатор для обучения

# odors_groups_valtest = [[0], [1], [2]]  # метки групп для объединения на валидации и тесте
odors_groups_valtest = [[i] for i in range(len(odors_unite))]  # В СЛУЧАЕ БЕЗ ГРУПП И ТЕСТА НА БО`ЛЬШИХ МЕТКАХ!
# например [[0, [1, 2], [3, 4]] - объединение ВВ (0), НС (1, 2) и НЕ ЦВ (3, 4), метки внутренние (индексация с нуля)

# weights = [0.5, 0.45, 1]  # веса для надклассификатора по группам НЕ ИСПОЛЬЗУЮТСЯ

unite = [[i] for i in range(len(odors_unite))]  # метки веществ (числовые), для объединения по группам на валидации
# unite = [[0], [1], [2], [3], [4], [5]]  # метки веществ (числовые), для объединения по группам на валидации

unite_test = [[i] for i in range(len(odors_unite))]  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте
# unite_test = [[0], [1], [2], [3], [4], [5]]  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте

# для конвертации приходящих меток. Сейчас делается автоматически в AbstractDataWorker
# labels_map = {1: 1, 2: 2, 4: 4} # for test
# labels_map = {8: 2, 16: 3, 32: 5}
# labels_map = {odors_unite[i][0]: i for i in range(len(odors_unite))}  # преобразование со степеней двойки на индексацию с нуля (авто)
# labels_map = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}  # преобразование со степеней двойки на индексацию с нуля
# labels_map = {4: 0, 16: 1, 1: 2}
# labels_map = {1: 0, 2: 0, 4: 1}

validationResult = {i: 0 for i in range(len(unite_test))}  # словарь по порядку, ключи в порядке, значения нули (авто)
# validationResult = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # словарь по порядку, ключи в порядке, значения нули

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

mixture_groups = [[i[0]] for i in odors_unite]
# mixture_groups = [[1], [8], [9], [16]]  # для классификации по клапанам (для Колиного)
# mixture_groups = [[odors_set[i]] for i in range(len(odors_set))]  # для классификации по клапанам (для Колиного) (авто) УБРАТЬ при обучении с объединением!
# mixture_groups = [[1], [2], [4], [8], [16], [32]]  # для классификации по клапанам (для Колиного) (авто) УБРАТЬ при обучении с объединением!
# mixture_groups = [[1, 2], [4]]  # тест перестановки для классификации по клапанам (для Колиного) при обучении с объединением!
# mixture_groups = [[1], [2], [4], [8], [16]]  # для классификации по клапанам (для Колиного) при обучении с объединением!
# mixture_groups = [[1, 4], [16, 8, 1]]          #для группировки. Пусть [4,5- Иное], [1,3- НС], [2,6 - ВВ]. Для Иное
# метка-'0', для НС метка-'1', для ВВ метка-'2'

validation_thresh = 80

acc_need = 80  # необходимая точность на валидации для успешного обучения
acc_need_two_ts = 75  # необходимая точность на двух последних валидациях для успешного обучения
acc_need_of_one_clf = 70  # необходимая точность от каждого классификатора в комитете

one_file = True  # костыль, если один файл для обучения и для тестирования, то True

rat_name = "2021_06_07_H78"  # для формирования отчёта

# обратить внимание на автоматически создаваемые переменные при серьезном изменении конфигурации!
