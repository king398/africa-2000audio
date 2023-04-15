import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import evaluate
import torch
from datasets import load_dataset, Audio
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
metric = evaluate.load("wer")

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")


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

    wer = metric.compute(predictions=pred_str, references=label_str, ) * 100
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
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


dataset = dataset.map(prepare_dataset, writer_batch_size=64, num_proc=32)
train_dataset = dataset['train']
valid_dataset = dataset['validation']
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False


class CFG:
    batch_size_per_device = 4
    epochs = 3
    train_steps = (int(49109 / (batch_size_per_device * 2))) * epochs
    eval_steps = (int(49109 / (batch_size_per_device * 2))) * 2


print(f"Training steps: {CFG.train_steps}")
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/mithil/PycharmProjects/africa-2000audio/model/whisper-small-baseline",
    # change to a repo name of your choice dsn_afrispeech
    per_device_train_batch_size=CFG.batch_size_per_device,
    learning_rate=1e-5,
    gradient_checkpointing=False,
    evaluation_strategy="steps",
    per_device_eval_batch_size=CFG.batch_size_per_device,  # try 4 and see if it crashes
    predict_with_generate=True,
    generation_max_length=448,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    num_train_epochs=CFG.epochs,
    # gradient_accumulation_steps=2,
    # deepspeed="/home/mithil/PycharmProjects/africa-2000audio/ds_config.json",

    seed=42,
    dataloader_num_workers=32,
    fp16=True,
    local_rank=os.environ["LOCAL_RANK"],
    eval_steps=CFG.eval_steps,
    save_steps=CFG.eval_steps,

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

processor.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()
trainer.train()
trainer.save_model(training_args.output_dir)
