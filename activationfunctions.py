# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
x = np.linspace(-10, 10, 100)
sigmoid = 1 / (1 + np.exp(-x))

plt.plot(x, sigmoid, label='Sigmoid')
plt.title("Sigmoid Function")
plt.grid(True)
plt.legend()
plt.show()

#Tanh Function
tanh = np.tanh(x)

plt.plot(x, tanh, label='Tanh', color='orange')
plt.title("Tanh Function")
plt.grid(True)
plt.legend()
plt.show()

#ReLU vs Leaky ReLU
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.01 * x)

plt.plot(x, relu, label='ReLU')
plt.plot(x, leaky_relu, label='Leaky ReLU', linestyle='--')
plt.title("ReLU vs Leaky ReLU")
plt.grid(True)
plt.legend()
plt.show()
