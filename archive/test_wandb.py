from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, HfArgumentParser
import wandb
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric



def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)



wandb.init()

parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]
training_args.report_to = "wandb"
print(training_args)

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = load_metric("glue", "mrpc")


model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()

for i in range(10):
    wandb.run.summary["test/"+str(i)] = i