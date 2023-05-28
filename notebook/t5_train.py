from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, T5TokenizerFast, T5ForConditionalGeneration, TrainingArguments
from datasets import load_dataset, load_metric

# Load tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained('t5-base')

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', max_length=512)
