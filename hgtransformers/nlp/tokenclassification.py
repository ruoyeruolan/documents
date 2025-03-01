# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : tokenclassification.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/24 20:10
@Description: 
"""

import evaluate
import numpy as np

from evaluate import EvaluationModule
from datasets import load_dataset, DatasetDict, Dataset, IterableDatasetDict, IterableDataset
from torch import mode
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer


def get_data(name):
    wnut: DatasetDict | Dataset | IterableDatasetDict | IterableDataset = load_dataset('wnut_17')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    # tokenized = tokenizer(wnut['train'][0]['tokens'], is_split_into_words=True)
    # tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
    return wnut, tokenizer


def tokenize_and_align_labels(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, inputs):

    tokenized = tokenizer(text=inputs['tokens'], is_split_into_words=True)

    labels = []
    for idx, label in enumerate(inputs[f"ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized['labels'] = labels
    return tokenized

def compute_metrics(p, label_list, seqeval: EvaluationModule):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list(p) for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    } if results else None


def main():
    data, tokenizer = get_data('wnut')
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    seqeval = evaluate.load('seqeval')

    tokenized = data.map(
        lambda x: tokenize_and_align_labels(tokenizer, x), batched=True
    )
    label_list = data["train"].features[f"ner_tags"].feature.names

    id2label = {
        0: "O",
        1: "B-corporation",
        2: "I-corporation",
        3: "B-creative-work",
        4: "I-creative-work",
        5: "B-group",
        6: "I-group",
        7: "B-location",
        8: "I-location",
        9: "B-person",
        10: "I-person",
        11: "B-product",
        12: "I-product",
    }
    label2id = {
        "O": 0,
        "B-corporation": 1,
        "I-corporation": 2,
        "B-creative-work": 3,
        "I-creative-work": 4,
        "B-group": 5,
        "I-group": 6,
        "B-location": 7,
        "I-location": 8,
        "B-person": 9,
        "I-person": 10,
        "B-product": 11,
        "I-product": 12,
    }

    model = AutoModelForTokenClassification.from_pretrained(
        'distilbert/distilbert-base-uncased', 
        num_labels=13,
        id2label=id2label, 
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir='token_classification',
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

    def compute_metrics_wrapper(p):
        return compute_metrics(p, label_list=label_list, seqeval=seqeval)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_wrapper, 
    )

    trainer.train()

