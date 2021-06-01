from watchdog.conrtollers.QtWatchdogController import QtWatchdogController
from configs.watchdog_config import is_train
import os
import datetime
import shutil


def clearLogs():
    currentTime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")  # получаем текущее время
    backupPath = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                 'watchdog', 'utils', 'backupLogs', currentTime)
    os.makedirs(backupPath, mode=0o777, exist_ok=True)  # создаём папку бэкапа с промежутками и без ошибок
    logs = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'logs')  # получаем и перемещаем логи и модели ниже
    if (os.path.exists(logs)):
        shutil.move(logs, backupPath)
    model1 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pretrained_model_LDA-ORS.mdl')
    if (os.path.exists(model1)):
        shutil.move(model1, backupPath)
    model2 = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'pretrained_model_SLP-ORS.mdl')
    if (os.path.exists(model2)):
        shutil.move(model2, backupPath)
    super_best = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'super_best_weights.weight')
    if (os.path.exists(super_best)):
        shutil.move(super_best, backupPath)
    # print(logs)
    # print(model1)
    # print(model2)


if __name__ == "__main__":
    if is_train:
        clearLogs()
    controller = QtWatchdogController()
