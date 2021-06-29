import numpy as np
import matplotlib.pyplot as plt
N_CHANNELS = 18
f = np.fromfile("C:\\WatchdogFiles\\input\\train_20210526_11_59_15__1_val.dat", "i2")
# s = np.fromfile("/home/quantum/Documents/watchdog_files/out/20210626_13_41_58/train_20210622_13_23_52_no_corr__1_val.dat", "i2")
# f = np.fromfile("/home/quantum/Documents/watchdog_files/out/20210626_09_13_25/train_20210622_13_23_52_no_corr_changed_22-06-2021_18-012_val.dat", "i2")
# print(f.shape)
# print(s.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# data_s = np.reshape(s, (-1, N_CHANNELS))
#
# print(len([i for i in data_s[:, -1] if i != 0]))
# print(len([i for i in data_f[:, -1] if i != 0]))
# print(data_f.shape)
# print(data_s.shape)
# data[:, -1] = np.where(data[:, -1] == 2, 16, data[:, -1])
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
# plt.plot(data_s[:,-1])
plt.plot(data[:,-1])
plt.show()