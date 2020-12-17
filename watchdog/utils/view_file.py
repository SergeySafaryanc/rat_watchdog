import numpy as np
import matplotlib.pyplot as plt
N_CHANNELS = 14
f = np.fromfile("C:\\WatchdogFiles\\input\\corr_ch_20201209_123241_N54_1series_train .dat", "i2")
print(f.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
plt.plot(data[:,-1])
plt.show()