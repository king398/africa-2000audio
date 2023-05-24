import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import torch
from datasets import load_dataset, Audio
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
print(dataset)
metric = evaluate.load("wer")


class CFG:
    batch_size_per_device = 16
    epochs = 3
    train_steps = (int(52161 / (batch_size_per_device * 2))) * epochs
    model = "openai/whisper-medium.en"


feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG.model)
tokenizer = WhisperTokenizer.from_pretrained(CFG.model, )
processor = WhisperProcessor.from_pretrained(CFG.model)


def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"], truncation=True, max_length=448).input_ids
    return batch


class ProgressLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"Progress: Evaluation step {state.global_step} of total {state.max_steps} steps.")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    print(pred_str[:5])
    print(label_str[:5])

    wer = metric.compute(predictions=pred_str, references=label_str) * 100
    print("WER: ", wer)

    return {"wer": wer}


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyway
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def normalize_text(batch):
    batch["transcription"] = batch["transcription"]
    return batch

dataset = dataset.map(prepare_dataset, writer_batch_size=64, num_proc=32)