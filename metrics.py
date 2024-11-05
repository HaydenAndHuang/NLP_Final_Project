"""Placeholder for metrics."""
from functools import partial
import evaluate
import numpy as np
import torch
# import torchmetrics.retrieval as retrieval_metrics

import pandas as pd


# CAPTIONING METRICS
def bleu(predictions, ground_truths, order):
    bleu_eval = evaluate.load("bleu")
    return bleu_eval.compute(
        predictions=predictions, references=ground_truths, max_order=order
    )["bleu"]       #  Maximum n-gram order to use when computing BLEU score. Defaults to 4.

def meteor(predictions, ground_truths):
    # https://github.com/huggingface/evaluate/issues/115
    meteor_eval = evaluate.load("meteor")
    return meteor_eval.compute(predictions=predictions, references=ground_truths)[
        "meteor"
    ]


def rouge(predictions, ground_truths):
    rouge_eval = evaluate.load("rouge")
    return rouge_eval.compute(predictions=predictions, references=ground_truths)[
        "rougeL"
    ]


# def vocab_diversity(predictions, references):
#     train_caps_tokenized = [
#         train_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
#         for train_cap in references
#     ]
#     gen_caps_tokenized = [
#         gen_cap.translate(str.maketrans("", "", string.punctuation)).lower().split()
#         for gen_cap in predictions
#     ]
#     training_vocab = Vocabulary(train_caps_tokenized, min_count=2).idx2word
#     generated_vocab = Vocabulary(gen_caps_tokenized, min_count=1).idx2word

#     return len(generated_vocab) / len(training_vocab)


# def vocab_novelty(predictions, tr_ground_truths):
#     predictions_token, tr_ground_truths_token = [], []
#     for gen, ref in zip(predictions, tr_ground_truths):
#         predictions_token.extend(gen.lower().replace(",","").replace(".","").split())
#         tr_ground_truths_token.extend(ref.lower().replace(",","").replace(".","").split())

#     predictions_vocab = set(predictions_token)
#     new_vocab = predictions_vocab.difference(set(tr_ground_truths_token))
    
#     vocab_size = len(predictions_vocab)
#     novel_v = len(new_vocab) / vocab_size
#     return vocab_size, novel_v

# def caption_novelty(predictions, tr_ground_truths):
#     unique_pred_captions = set(predictions)
#     unique_train_captions = set(tr_ground_truths)

#     new_caption = unique_pred_captions.difference(unique_train_captions)
#     novel_c = len(new_caption) / len(unique_pred_captions)
#     return novel_c

def type_token_ration(predictions):
    tokens = [word for text in predictions for word in text.split()]
    # Count unique words (types)
    unique_words = set(tokens)
    # Calculate TTR
    ttr = len(unique_words) / len(tokens) if tokens else 0
    return ttr


if __name__ == "__main__":
    df = pd.read_csv('musiccaps-baseline-expert.csv')
    novice_list = list(df['novice'])
    predictions = list(df['gen_expert'])
    ground_truths = list(df['caption'])

    results = {
        "bleu1": bleu(predictions, ground_truths, order=1),
        "bleu2": bleu(predictions, ground_truths, order=2),
        "bleu3": bleu(predictions, ground_truths, order=3),
        "bleu4": bleu(predictions, ground_truths, order=4),
        "meteor_1.0": meteor(predictions, ground_truths),
        "rougeL": rouge(predictions, ground_truths),
        "TTR": type_token_ration(predictions),
    }

    print(results)

    # {'bleu1': 0.25704510738713243, 'bleu2': 0.12273664637607094, 'bleu3': 0.06160973462259854, 'bleu4': 0.032996697123756684, 'meteor_1.0': 0.22219421581595938, 'rougeL': 0.23211966437052559, 'TTR': 0.0365224208898665}