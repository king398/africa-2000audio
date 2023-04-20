import os
import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Union

import evaluate
import numpy as np
import torch
from datasets import load_dataset, Audio
from transformers import Wav2Vec2CTCTokenizer, Seq2SeqTrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, \
    Trainer, \
    Wav2Vec2ForCTC

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
metric = evaluate.load("wer")
chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'


# tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-xls-r-300m", unk_token="[UNK]",
# pad_token="[PAD]", word_delimiter_token="|")


def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_remove_regex, '', batch["transcription"]).lower()
    return batch


def extract_all_chars(batch):
    all_text = " ".join(batch["transcription"])
    # print(all_text)
    vocab = list(set(all_text))
    # print(vocab)
    return {"vocab": [vocab], "all_text": [all_text]}


dataset = dataset.map(remove_special_characters)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
vocab_train = train_dataset.map(extract_all_chars, keep_in_memory=True, batched=True, batch_size=-1,
                                remove_columns=dataset["train"].column_names)
vocab_validation = validation_dataset.map(extract_all_chars, keep_in_memory=True,
                                          remove_columns=dataset["validation"].column_names)
vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_validation["vocab"][0][0]))
vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
import json

with open('/home/mithil/PycharmProjects/africa-2000audio/data/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)
# print(vocab_dict)
tokenizer = Wav2Vec2CTCTokenizer("/home/mithil/PycharmProjects/africa-2000audio/data/vocab.json", unk_token="[UNK]",
                                 pad_token="[PAD]", word_delimiter_token="|")
feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True,
                                             return_attention_mask=True)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch


train_dataset = train_dataset.map(prepare_dataset, writer_batch_size=64,
                                  num_proc=32)
validation_dataset = validation_dataset.map(prepare_dataset,
                                            writer_batch_size=64, num_proc=32)


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}



model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-xls-r-300m",
    attention_dropout=0.0,
    hidden_dropout=0.0,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.0,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer), )

model.freeze_feature_extractor()


class CFG:
    batch_size_per_device = 8
    epochs = 5
    train_steps = (int(49109 / (batch_size_per_device * 2))) * epochs


training_args = Seq2SeqTrainingArguments(
    output_dir="/home/mithil/PycharmProjects/africa-2000audio/model/wav2vec2-xls-r-300m-baseline-5-epoch",
    # change to a repo name of your choice dsn_afrispeech
    per_device_train_batch_size=4,
    gradient_checkpointing=True,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=CFG.batch_size_per_device,  # try 4 and see if it crashes
    predict_with_generate=True,
    # generation_max_length=448,
    report_to=["tensorboard", "wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    num_train_epochs=CFG.epochs,
    gradient_accumulation_steps=2,

    seed=42,
    fp16=True,
    local_rank=os.environ["LOCAL_RANK"],
    logging_steps=100,
    save_strategy="epoch",
    fp16_full_eval=True,
    learning_rate=1e-5,
    warmup_steps=100,
    group_by_length=True,
)

trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=processor.feature_extractor,
)
trainer.train()
