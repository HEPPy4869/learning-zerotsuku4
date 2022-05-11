import numpy as np

seed = 0
Q = 0
np.random.seed(seed)

for n in range(1, 11):
    reward = np.random.rand()
    Q = Q + (reward - Q) / n
    print(Q)