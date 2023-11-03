import json
import torch
import numpy as np
import pickle
from condbert import CondBertRewriter
from masked_token_predictor_bert import MaskedTokenPredictorBert
from transformers import BertTokenizer, BertForMaskedLM, BertTokenizerFast



# Load the dictionary from the text file
with open('../data/word_toxicity_scores.txt', 'r') as file:
    word_toxicity_scores = json.loads(file.read())


# Load the toxic_words_in_sentences list from the text file
with open('../data/toxic_words_in_sentences.txt', 'r') as file:
    toxic_words_in_sentences = json.loads(file.read())

# Load the non_toxic_words_in_sentences list from the text file
with open('../data/non_toxic_words_in_sentences.txt', 'r') as file:
    non_toxic_words_in_sentences = json.loads(file.read())


# Load the log_odds_ratios array from the text file
log_odds_ratios = np.loadtxt('../data/log_odds_ratios.txt')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
masked_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

def adjust_logits(logits, label=0):
    return logits - log_odds_ratios * 100 * (1 - 2 * label)

predictor = MaskedTokenPredictorBert(masked_model, tokenizer, max_len=250, device=device, label=0,
                                     contrast_penalty=0.0, logits_postprocessor=adjust_logits)

editor = CondBertRewriter(
    model=masked_model,
    tokenizer=tokenizer,
    device=device,
    neg_words=toxic_words_in_sentences,
    pos_words=non_toxic_words_in_sentences,
    word2coef=word_toxicity_scores,
    token_toxicities=log_odds_ratios,
    predictor=predictor,
)


# Save the editor object
with open('editor.pkl', 'wb') as file:
    pickle.dump(editor, file)