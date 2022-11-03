import numpy as np

#HW 2.2
#[print(i**2) for i in range(100)]
#[print(i**2) for i in range(100) if i % 2 == 0]


#HW 2.3
"""def gen(num, word):
    for j in range(1, num+1):
        yield word * j
    print(" ")

for n in gen(5,'Meow '):
    print(n)
"""

#HW 2.4
"""x= np.random.normal(loc=0.0, scale=1.0, size=(5, 5))
x = (np.where(x < 0.09, x**2, 42))
print(x[:, 3])"""

#Math
def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return s,ds