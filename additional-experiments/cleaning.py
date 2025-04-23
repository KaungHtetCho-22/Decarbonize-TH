import pandas as pd

path = '/home/multi-cam17/Documents/carbon/data/raw/owid-co2-data_mod_clean.csv'
df = pd.read_csv(path)
print(df)
df = df[df['year']>1949]
print(df)
print(df.isnull().sum(axis=0).sum())
# df.to_csv('after_1950.csv', index=False)


print(df.columns)