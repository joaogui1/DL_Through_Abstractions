import csv
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("circular.csv")

plt.scatter(df['x'], df['y'], c=df['z'])
plt.show()
