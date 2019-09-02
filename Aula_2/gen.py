import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

x_pluses = list(range(5, 51))
shuffle(x_pluses)

x_minuses = list(range(-50, -2))
shuffle(x_minuses)

y_pluses = range(5, 51)
y_minuses = range(-50, -2)

X = np.array([[x, y, z] for x, y, z in zip(x_pluses, y_pluses, np.ones(len(x_pluses)))] + [[x, y, -z] for x, y, z in zip(x_minuses, y_minuses, np.ones(len(x_pluses)))])

np.savetxt("linsep.csv", X, delimiter=",")
