import pandas as pd

df = pd.read_csv('MINISP_FULL_NATIVE.csv')
print(len(df))
print(len(df)-600-2-32)
print(len(df)%(600-2))
