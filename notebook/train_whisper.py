import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
import torch
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import WhisperTokenizer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, get_linear_schedule_with_warmup, AdamW

warnings.filterwarnings("ignore")
dataset = load_dataset("audiofolder",
                       data_dir="/home/mithil/PycharmProjects/africa-2000audio/data/train_hf",
                       )

dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
metric = evaluate.load("wer")


class CFG:
    batch_size_per_device = 4
    epochs = 4
    train_steps = (int(52161 / (batch_size_per_device * 2))) * epochs
    eval_steps = int(52161 / (batch_size_per_device * 2))
    model = "openai/whisper-large-v2"
    output_dir = "/home/mithil/PycharmProjects/africa-2000audio/model/whisper-large-4epoch-1e-5-adam"
    learning_rate = 1e-5
    evaluation_strategy = "epoch"
    predict_with_generate = True
    generation_max_length = 448
    report_to = ["tensorboard", "wandb"]
    greater_is_better = False
    push_to_hub = True
    gradient_accumulation_steps = 4
    fp16 = True
    # fp16_full_eval = True
    seed = 42
    dataloader_num_workers = 32
    logging_steps = 100
    save_strategy = "epoch"
    dataloader_pin_memory = True

    save_total_limit = 4
    remove_unused_columns = False
    # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names = ["labels"]
    tf32 = True
    gradient_checkpointing = True
    deepspeed = "ds_config.json"
    load_best_model_at_end = True
    metric_for_best_model = "wer"
    warmup_steps = 100


feature_extractor = WhisperFeatureExtractor.from_pretrained(CFG.model)
tokenizer = WhisperTokenizer.from_pretrained(CFG.model, language="english", task="transcribe")
processor = WhisperProcessor.from_pretrained(CFG.model, language="english", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["transcription"], truncation=True, max_length=448).input_ids
    return batch


def prepare_dataset_2(batch):
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
    print(f"Prediction: {pred_str[:2]}")
    print(f"Reference: {label_str[:2]}")

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
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


dataset['train'] = dataset['train'].map(prepare_dataset, writer_batch_size=64, num_proc=32,
                                        cache_file_name="cache/train_hf_cache.arrow")
dataset["train"] = dataset["train"].shuffle(seed=42)
dataset['validation'] = dataset['validation'].map(prepare_dataset, writer_batch_size=64, num_proc=32,
                                                  cache_file_name="cache/val_hf_cache.arrow")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
model = WhisperForConditionalGeneration.from_pretrained(CFG.model)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=CFG.output_dir,
    per_device_train_batch_size=CFG.batch_size_per_device,
    learning_rate=CFG.learning_rate,
    evaluation_strategy=CFG.evaluation_strategy,
    per_device_eval_batch_size=CFG.batch_size_per_device,
    predict_with_generate=CFG.predict_with_generate,
    generation_max_length=CFG.generation_max_length,
    report_to=CFG.report_to,
    greater_is_better=CFG.greater_is_better,
    push_to_hub=CFG.push_to_hub,
    gradient_accumulation_steps=CFG.gradient_accumulation_steps,
    fp16=CFG.fp16,
    #  fp16_full_eval=CFG.fp16_full_eval,
    seed=CFG.seed,
    dataloader_num_workers=CFG.dataloader_num_workers,
    logging_steps=CFG.logging_steps,
    save_strategy=CFG.save_strategy,
    dataloader_pin_memory=CFG.dataloader_pin_memory,
    save_total_limit=CFG.save_total_limit,
    remove_unused_columns=CFG.remove_unused_columns,
    label_names=CFG.label_names,
    num_train_epochs=CFG.epochs,
    tf32=CFG.tf32,
    gradient_checkpointing=CFG.gradient_checkpointing,
    deepspeed=CFG.deepspeed,
    load_best_model_at_end=CFG.load_best_model_at_end,
    metric_for_best_model=CFG.metric_for_best_model,
    warmup_steps=CFG.warmup_steps,

)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
    # optimizers=(optimizer, lr_scheduler),

)

model.config.use_cache = False
# trainer.optimizer = adam_bnb_optim
processor.save_pretrained(training_args.output_dir)
torch.cuda.empty_cache()
trainer.train()
trainer.save_model(training_args.output_dir)
repo_id = f"{CFG.output_dir.split('/')[-1]}"
model.push_to_hub(repo_id, use_auth_token=True, private=True)
processor.push_to_hub(repo_id, use_auth_token=True, private=True)
