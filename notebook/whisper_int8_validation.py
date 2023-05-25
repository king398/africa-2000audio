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
from peft import prepare_model_for_int8_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from peft import PeftConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
metric = evaluate.load("wer")
model_id = "model/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-full-fp16-training"  # Use the same model ID as before.

model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
).cuda()

processor = WhisperProcessor.from_pretrained(model_id,
                                             )
tokenizer = WhisperTokenizer.from_pretrained(model_id,
                                             )
feature_extractor = processor.feature_extractor
def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"], truncation=True, max_length=448).input_ids
    return batch


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
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


model.eval()
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

dataset['validation'] = dataset['validation'].map(prepare_dataset, writer_batch_size=64, num_proc=32,
                                                  cache_file_name="val_hf_cache.arrow")
valid_loader = DataLoader(dataset['validation'], batch_size=8,
                          collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor), pin_memory=True)
from tqdm import tqdm
import numpy as np

preds = None
labels_final = None
for step, batch in enumerate(tqdm(valid_loader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = (
                model.generate(
                    batch["input_features"].to("cuda"),


                )
                .cpu()
                .numpy()
            )
            labels = batch["labels"].cpu().numpy()
            labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
            decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
            wer = metric.compute(predictions=decoded_preds, references=decoded_labels)
            if preds is None:
                preds = np.array(decoded_preds)
                labels_final = np.array(decoded_labels)
            else:
                preds = np.append(preds, decoded_preds)
                labels_final = np.append(labels_final, decoded_labels)
import pandas as pd

df = pd.DataFrame({"predictions": preds, "labels": labels_final})
df.to_csv("oof/predictions.csv")
print(f"WER: {metric.compute(predictions=preds, references=labels_final)}")
