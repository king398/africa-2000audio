import pandas as pd
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/mithil/PycharmProjects/africa-2000audio/data/train_metadata.csv')
df = df.rename(columns={'transcript': 'transcription'})


def convert_audio_path(audio_path):
    base_path = "/home/mithil/PycharmProjects/africa-2000audio/data"
    sub_path = audio_path.split("/", 2)[2]
    new_audio_path = f"{base_path}/{sub_path}"
    new_audio_path = new_audio_path.replace(".mp3", ".wav")
    return new_audio_path


df['audio_path_local'] = df['audio_path'].apply(convert_audio_path)
df['file_name'] = df['audio_path_local'].apply(lambda x: x.split("/")[-1])

train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
print(train_df['audio_path_local'].values)
train_df.to_csv('/home/mithil/PycharmProjects/africa-2000audio/data/train_metadata_split.csv', index=False)
test_df.to_csv('/home/mithil/PycharmProjects/africa-2000audio/data/test_metadata_split.csv', index=False)
