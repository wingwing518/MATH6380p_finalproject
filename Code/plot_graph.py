# importing the modules
import numpy as np
import matplotlib.pyplot as plt

# load data to be plotted
loss_path = '../Log/resnet50_nopretrain_accuracy_list.npy'
loss_npy = np.load(loss_path)

print(loss_npy.shape)

# data to be plotted
x = np.arange(1, 27) * 2
y = loss_npy

plt.ylim(0, 1)
# plotting
plt.title("Classification accuracy on validation set")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.plot(x, y, color="red")
plt.show()