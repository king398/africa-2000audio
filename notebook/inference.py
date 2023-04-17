import warnings

import pandas as pd
import torch
from datasets import load_dataset, Audio
from tqdm import tqdm
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration
from torch.utils.data import DataLoader, Dataset

submission_df = pd.read_csv("/home/mithil/PycharmProjects/africa-2000audio/data/SampleSubmission.csv")
id_dict = {}
for i in submission_df['ID']:
    id_dict.update({i: '""'})

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/submission",
                       )
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
dataset = dataset.remove_columns(
    ["user_ids", "accent", "age_group", "country", "nchars", 'audio_paths', 'duration', 'origin', 'domain', 'split',
     'audio_path_local', 'transcription'])

model_path = "/home/mithil/PycharmProjects/africa-2000audio/model/whisper-small-baseline"
tokenizer = WhisperTokenizer.from_pretrained(model_path, language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(model_path).to(torch.device("cuda:1"))
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False


class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_features = feature_extractor(sample["audio"]["array"], sampling_rate=sample["audio"]["sampling_rate"],
                                           return_tensors="pt").input_features
        # input_features = input_features.to(torch.device("cuda:1"))
        id = sample['ID']
        input_features = input_features.squeeze(0)
        return input_features, id

    def __len__(self):
        return len(self.dataset)


loader = DataLoader(AudioDataset(dataset["test"]), batch_size=8, shuffle=False, num_workers=8, pin_memory=True,
                    prefetch_factor=4)

for i, (input_feature, ID) in enumerate(tqdm(loader, total=len(loader))):
    input_feature = input_feature.to(torch.device("cuda:1"))

    predicted_ids = model.generate(input_feature)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    for id_one, transcript_one in zip(ID, transcription):
        id_dict[id_one] = transcript_one

sub_df = pd.DataFrame()
sub_df['ID'] = id_dict.keys()
sub_df['transcript'] = id_dict.values()
sub_df.to_csv("/home/mithil/PycharmProjects/africa-2000audio/submission/whisper-small-3epoch-english-only.csv", index=False)
