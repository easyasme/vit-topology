import numpy as np

lst = np.arange(1, 1000 + 1)

lst2 = np.random.choice(lst, 10, replace=False)

print(lst2)