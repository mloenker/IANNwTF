import numpy as np

x= np.random.normal(loc=0.0, scale=1.0, size=(5, 5))
x = (np.where(x < 0.09, x**2, 42))
print(x[:, 3])
