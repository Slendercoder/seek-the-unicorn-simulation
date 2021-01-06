import pandas as pd
from FRA import classify_region

df_humans = pd.read_csv('./Data/humans_tolerance0.csv')
df_humans['Region'] = df_humans['Region'].apply(lambda x: [int(y) for y in x[1:-1].split(',')])
df_humans['Category'] = df_humans['Region'].apply(lambda x: classify_region(x, 1))
df_humans.to_csv('./Data/humans_tolerance0.csv')
