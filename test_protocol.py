import argparse
from configs.watchdog_config import odors

# parser = argparse.ArgumentParser()
# parser.add_argument('file_csv', type=str)
# parser.add_argument('file_to', type=str)
# args = parser.parse_args()

# path_to_file = args.file_csv
path_to_file = '/home/quantum/data/05_11_2020/out/20201105_14_41_33/29.10.2020.N56.TD.1_test_filt-0,7hz_result.csv'
csv_file = [el.split(',')[-1][:-1] for el in open(file=path_to_file, encoding='utf-8').readlines()]

table = list(zip(csv_file, [i[0] for i in odors] * 50))

is_zv = True
result = {}

for i in table:
    print(i)
    if i[0] == i[1]:
        result.update({i[1]: result.get(i[1], 0) + 1})
    else:
        result.update({i[1]: result.get(i[1], 0)})


title = f"По ЦВ/Не ЦВ ({round(len(csv_file)/len(odors), 1)})"
with open('/home/quantum/data/05_11_2020/out/TD_N56.txt', 'a') as res:
    res.write(f'{title}: {str(result)}')
