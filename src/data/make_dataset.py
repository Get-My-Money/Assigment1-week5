import re
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.sparse import vstack
from transformers import pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast


df = pd.read_csv('filtered.tsv', sep='\t')
df = df.drop(['Unnamed: 0'], axis=1)


for i in df.index:
    if df.loc[i, 'trn_tox'] >= 0.5:
        # Swap 'reference' and 'translation'
        df.loc[i, 'reference'], df.loc[i, 'translation'] = df.loc[i, 'translation'], df.loc[i, 'reference']
        
        # Swap 'ref_tox' and 'trn_tox'
        df.loc[i, 'ref_tox'], df.loc[i, 'trn_tox'] = df.loc[i, 'trn_tox'], df.loc[i, 'ref_tox']

df.to_csv('processed_dataset.csv', index=False)


toxic_sentence = list(df['reference'][:460000])
nontoxic_sentence = list(df['translation'][:460000])
toxic_labels = [1 for i in range(len(toxic_sentence))]
nontoxic_labels = [0 for i in range(len(nontoxic_sentence))]


# Initialize the CountVectorizer with a limited number of features
vectorizer = CountVectorizer(max_features=100000)

# Vectorize the toxic and non-toxic sentences separately to save memory
X_toxic = vectorizer.fit_transform(toxic_sentence)
X_nontoxic = vectorizer.transform(nontoxic_sentence)

# Stack the vectorized sentences
X = vstack([X_toxic, X_nontoxic])

# Prepare the labels
labels = toxic_labels + nontoxic_labels

# Train the logistic regression model
lr_model = LogisticRegression()
lr_model.fit(X, labels)

# Get the feature names (words) and their weights from the logistic regression model
feature_names = vectorizer.get_feature_names_out()
feature_weights = lr_model.coef_[0]

# Create a dictionary mapping words to their toxicity scores (weights)
word_toxicity_scores = dict(zip(feature_names, feature_weights))


# Save the dictionary as a text file in JSON format
with open('word_toxicity_scores.txt', 'w') as file:
    file.write(json.dumps(word_toxicity_scores))


# Combine toxic and non-toxic sentences
sentences = toxic_sentence + nontoxic_sentence

# Initialize lists to store toxic and non-toxic words
toxic_words_in_sentences = []
non_toxic_words_in_sentences = []

# Iterate over sentences
for sentence in sentences:
    # Split the sentence into words, considering punctuation
    words = re.findall(r'\b\w+\b', sentence)
    
    # Compute the toxicity scores for the words
    scores = [word_toxicity_scores.get(word, 0) for word in words]
    
    # Compute the threshold only if scores is not empty
    if scores:
        t1 = max(0.2, max(scores)/2)
        t2 = min(-0.2, min(scores)/2)
    else:
        t1 = 0.2
        t2 = -0.2

    # Find the toxic and non-toxic words
    toxic_words = [word for word, score in zip(words, scores) if score > t1]
    non_toxic_words = [word for word, score in zip(words, scores) if score <= t2]
    
    toxic_words_in_sentences.extend(toxic_words)
    non_toxic_words_in_sentences.extend(non_toxic_words)

# Make words unique by converting lists to sets
toxic_words_in_sentences = list(set(toxic_words_in_sentences))
non_toxic_words_in_sentences = list(set(non_toxic_words_in_sentences))

# Save the toxic_words_in_sentences list as a text file in JSON format
with open('toxic_words_in_sentences.txt', 'w') as file:
    file.write(json.dumps(toxic_words_in_sentences))

# Save the non_toxic_words_in_sentences list as a text file in JSON format
with open('non_toxic_words_in_sentences.txt', 'w') as file:
    file.write(json.dumps(non_toxic_words_in_sentences))


tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Function to count token occurrences
def count_tokens(texts):
    counter = defaultdict(lambda: 1)
    for text in tqdm(texts):
        tokens = tokenizer.encode(text)
        for token in tokens:
            counter[token] += 1
    return counter

# Count tokens
toxic_counts = count_tokens(toxic_sentence)
nontoxic_counts = count_tokens(nontoxic_sentence)
        
# Calculate toxicity ratios
toxicity_ratios = [toxic_counts[i] / (nontoxic_counts[i] + toxic_counts[i]) for i in range(len(tokenizer.vocab))]
toxicity_ratios = np.array(toxicity_ratios)
log_odds_ratios = np.maximum(0, np.log(toxicity_ratios / (1 - toxicity_ratios)))

# discourage meaningless tokens
for token in ['.', ',', '-', ';', "'"]:
    token_id = tokenizer.encode(token)[1]
    log_odds_ratios[token_id] = 3
    
for token in ['you', 'the']:
    token_id = tokenizer.encode(token)[1]
    log_odds_ratios[token_id] = 0


# Save the log_odds_ratios array as a text file
np.savetxt('log_odds_ratios.txt', log_odds_ratios)
