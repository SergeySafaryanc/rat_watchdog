import numpy as np
import matplotlib.pyplot as plt
N_CHANNELS = 18
f = np.fromfile("C:\\WatchdogFiles\\train_20210427_16_43_51_no_corr    _val1.dat", "i2")
print(f.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# data[:, -1] = np.where(data[:, -1] == 2, 16, data[:, -1])
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
plt.plot(data[:,-1])
plt.show()