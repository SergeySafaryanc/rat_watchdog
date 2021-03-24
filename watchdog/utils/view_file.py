import numpy as np
import matplotlib.pyplot as plt
N_CHANNELS = 16
f = np.fromfile("C:\\WatchdogFiles\\input\\test_20210318_08_39_57 .dat", "i2")
print(f.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
plt.plot(data[:,-1])
plt.show()