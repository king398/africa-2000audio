import pandas as pd
import os

df = pd.read_csv('/home/mithil/PycharmProjects/africa-2000audio/data/train_hf/validation/metadata.csv')
print(df.columns)


def check_file(id):
    if not os.path.exists(f"/home/mithil/PycharmProjects/africa-2000audio/data/train_hf/validation/{id}"):
        return False
    else:
        return True


df['file_exists'] = df['file_name'].apply(check_file)
df = df[df['file_exists'] == True]
