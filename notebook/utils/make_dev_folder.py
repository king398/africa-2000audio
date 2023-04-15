import pandas as pd
import glob

df = pd.read_csv('/home/mithil/PycharmProjects/africa-2000audio/data/dev_metadata.csv')
import shutil


def convert_audio_path(audio_path):
    base_path = "/home/mithil/PycharmProjects/africa-2000audio/data"
    sub_path = audio_path.split("/", 2)[2]
    new_audio_path = f"{base_path}/{sub_path}"
    return new_audio_path


df['audio_path_local'] = df['audio_paths'].apply(convert_audio_path)

new_paths = []
ids = []
for path in df['audio_path_local'].values:
    id = path.split("/")[-1].split(".")[0]
    shutil.move(path, f"/home/mithil/PycharmProjects/africa-2000audio/data/dev_hf/{id}.wav")
    new_paths.append(f"/home/mithil/PycharmProjects/africa-2000audio/data/dev_hf/{id}.wav")
df['audio_path_local'] = new_paths
df['transcription'] = df['transcription'].apply(lambda x: x.replace(" ", ""))
df.to_csv('/home/mithil/PycharmProjects/africa-2000audio/data/dev_hf/dev_metadata_hf.csv', index=False)
