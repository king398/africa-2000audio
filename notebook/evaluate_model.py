import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from joblib import Parallel, delayed
from statistics import mean
import evaluate
import torch
from datasets import load_dataset, Audio
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
import jiwer
from torch.utils.data import Dataset, DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from tqdm.auto import tqdm

normalizer = BasicTextNormalizer()

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
metric = evaluate.load("wer")


class CFG:
    batch_size_per_device = 16
    epochs = 4
    train_steps = (int(49109 / (batch_size_per_device * 2))) * epochs
    model = "/home/mithil/PycharmProjects/africa-2000audio/model/whisper-medium-4epoch-1e-5-deepspeed"


tokenizer = WhisperTokenizer.from_pretrained(CFG.model, language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained(CFG.model, language="English", task="transcribe")

feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG.model)
model = WhisperForConditionalGeneration.from_pretrained(f"{CFG.model}/checkpoint-1466").to(torch.device("cuda:1"))
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
        transcription = sample["transcription"]
        return input_features, id, transcription

    def __len__(self):
        return len(self.dataset)


preds = []
transcript_list = []

loader = DataLoader(AudioDataset(dataset["validation"]), batch_size=8, shuffle=False, num_workers=8, pin_memory=True,
                    prefetch_factor=4, drop_last=False)
for i, (input_feature, ID, transcript) in enumerate(tqdm(loader, total=len(loader))):
    input_feature = input_feature.to(torch.device("cuda:1"))

    predicted_ids = model.generate(input_feature)
    transcription_model = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription_model[0])
    normalized = [t.lower() for t in transcription_model]
    print(normalized[0])
    print(transcript[0])
    preds.extend(transcription_model)
    transcript_list.extend(transcript)
    print("WER: ", jiwer.wer(transcript, transcription_model))
    print("Normalized WER: ", jiwer.wer(transcript, normalized))

print("WER: ", jiwer.wer(transcript_list, preds))
normalized_preds = [normalizer(t) for t in preds]
print("WER: ", jiwer.wer(transcript_list, normalized_preds))
