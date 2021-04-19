import numpy as np
import matplotlib.pyplot as plt
N_CHANNELS = 18
f = np.fromfile("C:\\WatchdogFiles\\input\\train_20210415_14_49_58_.dat", "i2")
print(f.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# data[:, -1] = np.where(data[:, -1] == 2, 16, data[:, -1])
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
plt.plot(data[:,-1])
plt.show()