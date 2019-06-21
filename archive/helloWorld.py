import matplotlib.pyplot as plt
import numpy as np

print('Hello world!')
x = np.linspace(0, 5, 10)
y = x ** 2
plt.plot(x, y, 'r', x, x ** 3, 'g', x, x ** 4, 'b')