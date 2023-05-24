import pandas as pd
import glob
import os

df = pd.read_csv('/home/mithil/PycharmProjects/africa-2000audio/data/valid_metadata_split.csv')
import shutil

os.makedirs('/home/mithil/PycharmProjects/africa-2000audio/data/train_hf/validation', exist_ok=True)


def move(path):
    id = path.split("/")[-1].split(".")[0]
    shutil.move(path, f"/home/mithil/PycharmProjects/africa-2000audio/data/train_hf/validation/{id}.wav")
    return f"/home/mithil/PycharmProjects/africa-2000audio/data/dev_hf/{id}.wav"


df["audio_path_local"] = df["audio_path_local"].apply(move)
print(df)
df.to_csv("/home/mithil/PycharmProjects/africa-2000audio/data/train_hf/validation/metadata.csv",index=False)