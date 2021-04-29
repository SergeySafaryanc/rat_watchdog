import os

from configs.watchdog_config import *

from loguru import logger

class Singleton(object):
    __instance = dict()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.__instance = super(Singleton, cls).__new__(cls)
        return cls.__instance

    @classmethod
    def set(cls, key, value):
        cls.__instance.update({key: value})

    @classmethod
    def get(cls, key):
        return cls.__instance.get(key, "")

    @classmethod
    def clear(cls):
        return cls.__instance.clear()

    @classmethod
    def items(cls):
        return cls.__instance.items()

    @classmethod
    def text(cls):
        return "\n".join([f"{k}: {v}" for k, v in cls.__instance.items()])

    def __repr__(self):
        return f"Singleton[{self.__instance}]"


def write(text=None, is_first=False, is_test=False):
    # TODO убрать костыль, сделав __exp_folder из AbstractDataWorker общим для проекта, например
    # это один большой костыль, совпадающий по коду с указанием переменной __exp_folder
    folders = [fname for fname in os.listdir(f"{out_path}{os.sep}") if
               os.path.isdir(os.path.join(f"{out_path}{os.sep}", fname))]
    # logger.info(folders)
    folders = sorted(folders, key=lambda x: os.stat(os.path.join(f"{out_path}{os.sep}", x)).st_mtime)
    # logger.info(folders)
    readme_exp_folder=f"{out_path}{os.sep}{folders[-1]}"
    while (len(os.listdir(readme_exp_folder)) == 0):
        folders = folders[:-1]
        readme_exp_folder = f"{out_path}{os.sep}{folders[-1]}"
    # logger.info(readme_exp_folder)
    # конец большого костыля
    file_to_record = f"{readme_exp_folder}{os.sep}{rat_name}.txt"
    stream = open(file=file_to_record, mode='a')
    if (os.path.getsize(file_to_record)):  # перенос строки в случае дозаписи
        stream.write('\n')
    stream.write(text)
    stream.close()
    Singleton.clear()  # очистка после вывода


if __name__ == '__main__':
    Singleton.set("vf", 1)
    print(Singleton.get("vf"))

    Singleton.set("cd", "Hello")
    print(Singleton.get("cd"))
