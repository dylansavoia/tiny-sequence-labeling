import torch
import numpy as np
import pickle, json
from training import SeqVocabulary, SeqModel

MODELSPATH = 'models/model.pth'
testphrase = 'search hello world on youtube'

base_name = MODELSPATH.rsplit('.', 1)[0]
vocab_file_name  = base_name + ".pkl"
params_file_name = base_name + ".json"

with open(vocab_file_name, 'rb') as f:
    vocabs = pickle.load(f)
toks_vocab = SeqVocabulary(vocabs[0])
lbls_vocab = SeqVocabulary(vocabs[1])

with open(params_file_name, 'r') as f:
    params = json.load(f)

model = SeqModel(params)
model.load_state_dict(
    torch.load(
        MODELSPATH,
        map_location=torch.device('cpu')
    )
)
model.eval()

out = model.classify(testphrase, toks_vocab, lbls_vocab)
print(testphrase)
print(out)
