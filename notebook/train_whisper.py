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


class CFG:
    batch_size_per_device = 4
    epochs = 3
    train_steps = (int(52161 / (batch_size_per_device * 2))) * epochs
    eval_steps = int(52161 / (batch_size_per_device * 2))
    model = "openai/whisper-large-v2"
    output_dir = "/home/mithil/PycharmProjects/africa-2000audio/model/whisper-large-v2-3epoch-1e-5-cosine-deepspeed-full-fp16-training"


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
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def normalize_text(batch):
    batch["transcription"] = batch["transcription"]
    return batch


dataset = dataset.map(normalize_text)
dataset['train'] = dataset['train'].map(prepare_dataset, writer_batch_size=64, num_proc=32,
                                        cache_file_name="cache/train_hf_cache.arrow")

dataset['validation'] = dataset['validation'].map(prepare_dataset, writer_batch_size=64, num_proc=32,
                                                  cache_file_name="cache/val_hf_cache.arrow")
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
model = WhisperForConditionalGeneration.from_pretrained(CFG.model)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

training_args = Seq2SeqTrainingArguments(
    output_dir=CFG.output_dir,
    # change to a repo name of your choice dsn_afrispeech
    per_device_train_batch_size=CFG.batch_size_per_device,
    learning_rate=1e-5,
    evaluation_strategy="epoch",
    per_device_eval_batch_size=CFG.batch_size_per_device,  # try 4 and see if it crashes
    predict_with_generate=True,
    generation_max_length=448,
    report_to=["tensorboard", "wandb"],
    greater_is_better=False,
    push_to_hub=False,
    gradient_accumulation_steps=4,
    fp16=True,
    fp16_full_eval=True,

    seed=42,
    dataloader_num_workers=32,
    logging_steps=100,
    save_strategy="epoch",
    dataloader_pin_memory=True,

    save_total_limit=1,
    remove_unused_columns=False,
    # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"],
    save_steps=CFG.eval_steps,
    num_train_epochs=CFG.epochs,
    tf32=True,
    gradient_checkpointing=True,
    deepspeed="ds_config.json",
    load_best_model_at_end=True,
    metric_for_best_model="wer",
)
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# This callback helps to save only the adapter weights and remove the base model weights.
class SavePeftModelCallback(TrainerCallback):
    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs,
    ):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback],
    # optimizers=(adam_bnb_optim, None),
    compute_metrics=compute_metrics,

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
