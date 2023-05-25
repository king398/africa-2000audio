import pandas as pd
import os
from transformers import (
    AutomaticSpeechRecognitionPipeline,
    WhisperForConditionalGeneration,
    WhisperTokenizer,
    WhisperProcessor,
)
from peft import PeftModel, PeftConfig
import torch
from tqdm import tqdm
import warnings
import datasets
from datasets import Audio
from typing import Any, Dict, List, Union
from dataclasses import dataclass

warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

language = "English"
task = "transcribe"
test_df = pd.read_csv("/home/mithil/PycharmProjects/africa-2000audio/data/test_metadata.csv")
test_df['audio'] = test_df['audio_paths'].replace('/AfriSpeech-100/',
                                                  '/home/mithil/PycharmProjects/africa-2000audio/data/',
                                                  regex=True)

dev_df = pd.read_csv("/home/mithil/PycharmProjects/africa-2000audio/data/dev_metadata.csv")
dev_df['audio'] = dev_df['audio_paths'].replace('/AfriSpeech-100/',
                                                '/home/mithil/PycharmProjects/africa-2000audio/data/',
                                                regex=True)

df = pd.concat([test_df, dev_df])
id_path_dict = dict(zip(df['ID'], df['audio']))

peft_model_id = "model/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-actual-3/checkpoint-9780/adapter_model"  # Use the same model ID as before.
peft_config = PeftConfig.from_pretrained(peft_model_id)

model = WhisperForConditionalGeneration.from_pretrained(
    peft_config.base_model_name_or_path, load_in_8bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_model_id)
processor = WhisperProcessor.from_pretrained("model/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-actual-3",
                                             language=language, task=task)
tokenizer = WhisperTokenizer.from_pretrained("model/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-actual-3",
                                             language=language, task=task)

feature_extractor = processor.feature_extractor
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
pipe = AutomaticSpeechRecognitionPipeline(model=model, tokenizer=tokenizer, feature_extractor=feature_extractor, )

dataset = datasets.Dataset.from_pandas(df=df)
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))


def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = torch.ones(448, dtype=torch.long)
    batch['ID'] = batch['ID']
    return batch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        ids = [feature["ID"] for feature in features]
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
        batch["ID"] = ids

        return batch


model.eval()
dataset = dataset.map(prepare_dataset, writer_batch_size=64, num_proc=32,
                      cache_file_name="test_hf_cache.arrow", )
valid_loader = DataLoader(dataset, batch_size=8,
                          collate_fn=DataCollatorSpeechSeq2SeqWithPadding(processor), pin_memory=True, shuffle=False)
submission = pd.DataFrame(columns=["ID", "transcript"])

for i, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            generated_tokens = model.generate(
                batch["input_features"].to("cuda"),
                ).cpu().numpy()
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            for id, pred in zip(batch["ID"], decoded_preds):
                submission = pd.concat([submission, pd.DataFrame({"ID": id, "transcript": pred}, index=[0])],
                                       ignore_index=True)

submission.to_csv("submission/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-actual-3.csv", index=False)
normalizer = BasicTextNormalizer()
submission["transcript"] = submission["transcript"].apply(normalizer)
submission.to_csv("submission/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-actual-3-cleaned.csv", index=False)
