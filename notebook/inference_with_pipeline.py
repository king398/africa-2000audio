import pandas as pd
import torch
from transformers import pipeline
from datasets import load_dataset, Audio
import glob
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
device = "cuda:1" if torch.cuda.is_available() else "cpu"
submission_df = pd.read_csv("/home/mithil/PycharmProjects/africa-2000audio/data/SampleSubmission.csv")
test_metadata = pd.read_csv("/home/mithil/PycharmProjects/africa-2000audio/data/submission/test/metadata.csv")
test_metadata['audio_path_local'] = test_metadata['audio_path_local'].replace('/home/mithil/PycharmProjects/africa-2000audio/data/dev_hf/', '/home/mithil/PycharmProjects/africa-2000audio/data/submission/test/', regex=True)

print(test_metadata.head())
id_dict = {}
for i in submission_df['ID']:
    id_dict.update({i: '""'})

pipe = pipeline(
    "automatic-speech-recognition",
    model="/home/mithil/PycharmProjects/africa-2000audio/model/whisper-small-baseline",
    chunk_length_s=30,
    device=device,
    stride_length_s=[6, 0],
)
files = test_metadata['audio_path_local'].tolist()
ID = test_metadata['ID'].tolist()
batch_size = 8
for idx in tqdm(range(0, len(files), batch_size)):
    # Get the current batch of audio files
    current_batch = files[idx:idx + batch_size]
    current_batch_id = ID[idx:idx + batch_size]
    # Process and transcribe each audio file in the current batch
    transcription = pipe(current_batch, batch_size=8,)
    for index, i in enumerate(transcription):
        id_dict.update({current_batch_id[index]: i['text']})


sub_df = pd.DataFrame()
sub_df['ID'] = id_dict.keys()
sub_df['transcript'] = id_dict.values()
sub_df.to_csv("/home/mithil/PycharmProjects/africa-2000audio/submission/whisper_baseline_with_pipeline.csv",
              index=False)
