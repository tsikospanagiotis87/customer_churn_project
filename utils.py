import pandas as pd
import numpy as np

def df_info(df):
    df_c = df.copy()
    info = pd.concat([df_c.dtypes, df_c.nunique()], axis= 1)
    info.columns = ['type', 'unique_values']
    return info

def entropy(df, y):
    entropy = 0
    for i in range(df[y].nunique()):
        entropy += - df[y].value_counts(normalize= True).values[i] * np.log2(df[y].value_counts(normalize= True).values[i])
    return entropy

def conditional_entropy(df, y, x):
    entropy = 0
    counter = 0 
    for i in range(df[x].nunique()):
        for j in range(df[y].nunique()):
            entropy += - df[x].value_counts(normalize= True).values[i] * df.groupby(x)[y].value_counts(normalize= True).values[counter] * np.log2(df.groupby(x)  [y].value_counts(normalize= True).values[counter])
            counter += 1
    return entropy
