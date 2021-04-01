import numpy as np
import matplotlib.pyplot as plt

N_CHANNELS = 18
f = np.fromfile("/home/quantum/Documents/watchdog_files/inp/30.03.21.N71.2.dat", "i2")
print(f.shape)
data = np.reshape(f, (-1, N_CHANNELS))
# fig, axes = plt.subplots(N_CHANNELS, 1, figsize=(16, 8));
# # for i, ax in enumerate(axes):
# axes.plot(data[-1])
plt.plot(data[:, -1])
plt.show()
