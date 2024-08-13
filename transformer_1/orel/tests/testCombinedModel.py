import pandas as pd


# Load the dataset
dataset_path = 'output.csv'
df = pd.read_csv(dataset_path)


def contains_only_hebrew(text):
    return all(('א' <= char <= 'ת') or char == ' ' for char in text)


# for index, row in df.iterrows():
#     if index > 10:
#         break
    
#     print(row[0])


# print(df.apply(contains_only_hebrew))
# print(df.value_counts().get('של', 0))
print(df.value_counts())
