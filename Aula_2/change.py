import numpy as np
import pandas as pd
from random import shuffle
import matplotlib.pyplot as plt

def f(x, y):
    return 1.0*np.sign(x*x + y*y - 1.1)

df = pd.read_csv("test.csv", names=['x', 'y'])

df['z'] = [f(x, y) for x, y in zip(df['x'], df['y'])]


# df.to_csv("circular.csv", index='false')

# X = np.array([[x, y, z] for x, y, z in zip(x_pluses, y_pluses, np.ones(len(x_pluses)))] + [[x, y, -z] for x, y, z in zip(x_minuses, y_minuses, np.ones(len(x_pluses)))])

# np.savetxt("linsep.csv", X, delimiter=",")
