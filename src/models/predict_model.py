import torch
import pickle
import numpy as np
import pandas as pd
from datasets import load_metric
from transformers import pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast


def adjust_logits(logits, label=0):
    return logits - log_odds_ratios * 100 * (1 - 2 * label)

# Load the editor object
with open('editor.pkl', 'rb') as file:
    editor = pickle.load(file)


dataset = pd.read_csv('../data/processed_dataset.csv')


X_test = list(dataset['reference'][460000:])
Y_test = list(dataset['translation'][460000:])
Y_test = [[sen] for sen in Y_test]
Y_pred = []


for sentence in X_test:
    Y_pred.append((editor.translate(sentence, prnt=False)))


metric = load_metric("sacrebleu")
results = metric.compute(predictions=Y_pred, references=Y_test)
print("sacrebleu metric:", results)


# Combine the toxic and non-toxic sentences and labels from test set
toxic_sentence, nontoxic_sentence = list(dataset['reference'][460000:]), list(dataset['translation'][460000:])
labels = [1 for i in range(len(toxic_sentence))] + [0 for i in range(len(nontoxic_sentence))]

# Create a pipeline with a TfidfVectorizer and LogisticRegression
pipeline = make_pipeline(TfidfVectorizer(max_features=100000), LogisticRegression())

# Train the model
pipeline.fit(toxic_sentence + nontoxic_sentence, labels)

# Predict the toxicity scores of the rewritten sentences
rewritten_toxicity_scores = pipeline.predict_proba(Y_pred)[:, 1]
  
# Predict the toxicity scores of the original toxic sentences
original_toxic_scores = pipeline.predict_proba(toxic_sentence)[:, 1]

# Predict the toxicity scores of the original non-toxic sentences
original_nontoxic_scores = pipeline.predict_proba(nontoxic_sentence)[:, 1]


average_toxic_score = original_toxic_scores.mean()
average_predicted_score = rewritten_toxicity_scores.mean()
average_nontoxic_score = original_nontoxic_scores.mean()

print(f"Average Toxicity Score for Toxic Sentences: {average_toxic_score}")
print(f"Average Toxicity Score for Non-Toxic Sentences: {average_nontoxic_score}")
print(f"Average Toxicity Score for Predicted Sentences: {average_predicted_score}")


final_untoxicity_score = (average_toxic_score - average_predicted_score)/(average_toxic_score - average_nontoxic_score)
print("Effectiveness of the detoxification:", final_untoxicity_score)

