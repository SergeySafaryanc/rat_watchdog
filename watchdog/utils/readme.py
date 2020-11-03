from configs.watchdog_config import *
import os

# from watchdog.worker.AbstractDataWorker import *
from watchdog.worker.AbstractDataWorker import ExpFolder


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
    def items(cls):
        return cls.__instance.items()

    @classmethod
    def text(cls):
        return "\n".join([f"{k}: {v}" for k, v in cls.__instance.items()])

    def __repr__(self):
        return f"Singleton[{self.__instance}]"


class Stats(ExpFolder):
    def __init__(self):
        super().__init__()
        path_to_file = ""
        csv_file = [el.split(';')[-1][:-1] for el in open(file=path_to_file, encoding='utf-8').readlines()]
        table = list(zip(csv_file, [i[0] for i in odors] * 50))
        result = {}

        for i in table:
            if i[0] == i[1]:
                result.update({i[1]: result.get(i[1], 0) + 1})
            else:
                result.update({i[1]: result.get(i[1], 0)})

        Singleton.set("Predict", str(result))
        # for k, v in result.items():
        #     print(f'{k} - {v}/{int(round(len(table) / len(result), 0))}')

# 29.10.2020.N56.2___res
# 29.10.2020.N56.2__res - [*[!_]_res]

#
# def stats(directory: str):
#     tmp = {}
#     if os.path.isdir(directory):
#         with open()
#     pass


def write(text=None, is_first=False, is_test=False):
    stream = open(file=f"{out_path}/{rat_name}.txt", mode='a')
    stream.write(text)
    stream.close()


if __name__ == '__main__':
    Singleton.set("vf", 1)
    print(Singleton.get("vf"))

    Singleton.set("cd", "Hello")
    print(Singleton.get("cd"))
