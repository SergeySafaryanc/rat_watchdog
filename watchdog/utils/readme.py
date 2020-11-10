from configs.watchdog_config import *


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


def write(text=None, is_first=False, is_test=False):
    stream = open(file=f"{out_path}/{rat_name}.txt", mode='a')
    stream.write(text)
    stream.close()


if __name__ == '__main__':
    Singleton.set("vf", 1)
    print(Singleton.get("vf"))

    Singleton.set("cd", "Hello")
    print(Singleton.get("cd"))
