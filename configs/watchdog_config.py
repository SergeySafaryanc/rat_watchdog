from configs.json_parser_configuration import Config
config_file = "/home/quantum/PycharmProjects/rat_watchdog_novosib/config.json"
config = Config(file=config_file)

# Socket conf

HOST = config.HOST
# HOST = '10.132.230.4'
PORT = config.PORT  # 5000 если передача удалённая, 4000 если с этого же устройства

# Worker conf В директорию out_path выводится файл: dat inf с названием train или test.py + время начала записи файлы
# с результатами имеют такое же название как и у dat файла с префиксами: result, _validation_result,
# _responses_classifiers
wait_time = config.wait_time  # sec # ожидание детекта
epoch_time = config.epoch_time  # sec # время перерыва чтения из файла
inp_path = "/home/quantum/Documents/watchdog_files/inp"  # для Windows
out_path = "/home/quantum/Documents/watchdog_files/out"  # для Windows
# inp_path = "~/WatchdogFiles/input"  # для Linux
# out_path = "~/WatchdogFiles/output"  # для Linux

sampling_rate = config.sampling_rate  # частота дискретизации
decimate_rate = config.decimate_rate  # прорежение данных, 10 для PLEXON онлайн, 1 для ТД1 и оффлайн (.dat)
num_channels = config.num_channels  # число каналов ЭГ+дыхание+метка

num_counter_for_refresh_animal = config.num_counter_for_refresh_animal  # количество стимулов для смены животного

count_train_stimuls = config.count_train_stimuls  # число тренировок
train_step = config.train_step  # число смещений (кол-во клапанов * 5)

data_source_is_file = config.data_source_is_file  # выбор датасорса, true - file, false - socket
is_result_validation = config.is_result_validation  # не трогать (типа валидация, кол-во правильных из скольких)

is_train = config.is_train  # обучение или тестирование
use_auto_train = config.use_auto_train  # обучение сразу после достижения count_train_stimuls (не трограть)

# Доп. запись
# odors = [("Мухамад 1", "#ffff00"), ("Гребеньков 1", "#ffff00"), ("Лежнев 1", "#ffff00")]
# odors = [("ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00"), ("НЕ ЦВ", "#ffff00")]
# odors = [("ТНТ", "#ffff00"), ("Приправа", "#ffff00"), ("Корвалол", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("Героин", "#ffff00"), ("Амфетамин", "#ffff00"), ("ТНТ", "#ffff00"),
#          ("октоген", "#ffff00"), ("тетрил", "#ffff00"), ("Гексоген 10-15", "#ffff00"), ("воздух", "#ffff00")]
# odors = [("ТНТ", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
# odors = [("ТНТ", "#ffff00"), ("КП", "#ffff00"), ("Воздух", "#ffff00")]
odors = config.odors  # во втором случае OK на английском
# odors = [("Обнаружено ЦВ!", "#ffff00"), ("ОК", "#ffff00"), ("OK", "#ffff00")]  # во втором случае OK на английском

# ("Вертилов 2", "#ffff00"),
# ("Губа 2", "#ffff00"), ("Чуб 2", "#ffff00"),
# ("5", "#ffff00"), ("3", "#ffff00")
# ]  # Расшифровка и цвет вещества (на экран и в таблицу)
# odors_set = [1, 2, 4, 8, 16]  # метки веществ с клапанов
# odors_set = [1, 8, 9, 16]  # метки веществ с клапанов
odors_set = config.odors_set # метки веществ с клапанов, передаётся в классификатор для обучения
# odors_set = [1, 2, 4]  # метки веществ с клапанов, передаётся в классификатор для обучения
weights = config.weights  # веса для надклассификатора: ТНТ=1, КП=0.9, Воздух=0.4
# weights = [1, 0.9, 0.51]  # веса для надклассификатора: ТНТ=1, КП=0.9, Воздух=0.4

# unite = [[0], [1], [2], [3], [4]]  # метки веществ (числовые), для объединения по группам на валидации
# unite = [[0], [1, 2, 3, 4]]  # метки веществ (числовые), для объединения по группам на валидации
# unite = [[0], [1], [2], [3]]  # метки веществ (числовые), для объединения по группам на валидации
unite = config.unite  # метки веществ (числовые), для объединения по группам на валидации
# unite = [[0], [1], [2]]  # метки веществ (числовые), для объединения по группам на валидации

# unite_test = [[0], [1], [2], [3], [4]]  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте
# unite_test = [[0], [1], [2], [3]]  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте
unite_test = config.unite_test  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте
# unite_test = [[0], [1], [2]]  # Для подсчета результатов при смене клапанов, для вывода по группам на тесте

# #Для Косенко
# num_channels = 18  # число каналов ЭГ+дыхание+метка
# count_train_stimuls = 75  # число тренировок
# train_step = 15  # число смещений (кол-во клапанов * 5)
# is_train = False  # обучение или тестирование
# is_train = True  # обучение или тестирование
# odors = [("Ам. селитра", "#ffff00"),
#          ("Тетрил", "#ffff00"),
#          ("Гексоген", "#ffff00"),
#          ("героин", "#ffff00"),
#          ("ТЭН", "#ffff00")]  # Расшифровка и цвет вещества
# odors_set = [1, 2, 4]  # метки веществ с клапанов, передаётся в классификатор для обучения
# unite = [[0, 1], [2]]  # метки веществ (числовые) для объединения
# unite_test = [[0, 1], [2]]  # Для подсчета результатов при смене клапанов, для вывода результата
# one_file = False  # костыль, если один файл для обучения и для тестирования, то True
# rat_name = "16.06.2020 Н50"  # для формирования отчёта
# labels_map = {4: 0, 16: 1, 1: 2}

# для конвертации приходящих меток
# labels_map = {1: 1, 2: 2, 4: 4} # for test
# labels_map = {8: 2, 16: 3, 32: 5}
labels_map = {odors_set[i]: i for i in range(len(odors_set))}  # преобразование со степеней двойки на индексацию с нуля (авто)
# labels_map = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5}  # преобразование со степеней двойки на индексацию с нуля
# labels_map = {4: 0, 16: 1, 1: 2}

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
# mixture_groups = [[1], [8], [9], [16]]  # для классификации по клапанам (для Колиного)
mixture_groups = [[odors_set[i]] for i in range(len(odors_set))]  # для классификации по клапанам (для Колиного) (авто) УБРАТЬ при обучении с объединением!
# mixture_groups = [[1, 2], [4]]  # тест перестановки для классификации по клапанам (для Колиного) при обучении с объединением!
# mixture_groups = [[1], [2], [4], [8], [16]]  # для классификации по клапанам (для Колиного) при обучении с объединением!
# mixture_groups = [[1, 4], [16, 8, 1]]          #для группировки. Пусть [4,5- Иное], [1,3- НС], [2,6 - ВВ]. Для Иное
# метка-'0', для НС метка-'1', для ВВ метка-'2'
validation_thresh = 80

# обратить внимание на автоматически создаваемые переменные при серьезном изменении конфигурации!

acc_need = 80  # необходимая точность на валидации для успешного обучения
acc_need_two_ts = 75  # необходимая точность на двух последних валидациях для успешного обучения
acc_need_of_one_clf = 70  # необходимая точность от каждого классификатора в комитете

one_file = False  # костыль, если один файл для обучения и для тестирования, то True

rat_name = config.rat_name  # для формирования отчёта
