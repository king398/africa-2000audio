from datasets import load_dataset, Audio
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf", streaming=True,
                       )
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small", language="English", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="English", task="transcribe")
metric = evaluate.load("wer")


def prepare_dataset(batch):
    audio = batch["audio"]
    print(audio)

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"]).input_ids
    return batch


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

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


dataset = dataset.map(prepare_dataset)

train_dataset = dataset['train']
valid_dataset = dataset['validation']

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
training_args = Seq2SeqTrainingArguments(
    output_dir="/home/mithil/PycharmProjects/africa-2000audio$/model/whisper-small-baseline",
    # change to a repo name of your choice dsn_afrispeech
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,  # try 4 and see if it crashes
    predict_with_generate=True,
    generation_max_length=225,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    num_train_epochs=5,
    push_to_hub=False,
    torch_compile=True,
    seed=42
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor, )

processor.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()
trainer.train()
trainer.save_model(training_args.output_dir)
