from configs.watchdog_config import *
from os.path import isfile


def build_protocol(values=None, is_first=False):
    file_name = f'/{rat_name}.txt'
    stream = open(file=out_path + file_name, mode='a')
    if is_first:
        stream.write(f'Rat: {rat_name}\nNum channels: {num_of_channels}\nTrain step: {train_step}\n')
    else:
        stream.write(f'Val: {str([round(i, 2) for i in values])} - [MAX={max(values)}, MIN={min(values)}, AVG={round(sum(values)/len(values), 2)}]')
    stream.close()


