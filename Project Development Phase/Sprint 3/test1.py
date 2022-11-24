import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.pandas.set_option('display.max_columns', None)
dataset = pd.read_csv('static/dataset/train.csv')
dat=dataset.head()
print(dat)
