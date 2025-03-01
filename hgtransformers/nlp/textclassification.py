# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : textclassification.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/20 23:55
@Description: 
"""

from huggingface_hub import notebook_login, login
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

import evaluate
import numpy as np

# notebook_login()

imdb = load_dataset('imdb')

tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')


def preprocess_function(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, example: dict):
    return tokenizer(example['text'], truncation=True)


# tokenizered_imdb = imdb.map(preprocess_function, batched=False)

tokenized_imdb = imdb.map(lambda x: preprocess_function(tokenizer, x), batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

accuracy = evaluate.load(path='accuracy')

def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    'distilbert/distilbert-base-uncased', 
    num_labels=2,
    id2label = id2label,
    label2id = label2id
)

train_args = TrainingArguments(
    output_dir='awesome_model',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=.01,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_imdb['train'],
    eval_dataset=tokenized_imdb['text'],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)