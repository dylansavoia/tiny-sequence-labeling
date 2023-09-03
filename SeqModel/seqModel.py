import torch
from torch import nn
import pickle, json, os
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from .seqVocabulary import SeqVocabulary

class SeqModel(nn.Module):
    def __init__(self, params={}, load="model", dev=torch.device('cpu')):
        super(SeqModel, self).__init__()
        self.dev = dev

        isPretrained = (params == {})

        # Load pre-trained model params for model setup
        if isPretrained:
            modelPath  = f"{load}.pth"
            vocabPath  = f"{load}.pkl"
            paramsPath = f"{load}.json"

            KILL(not os.path.isfile(modelPath), f"Model not found: {modelPath}")
            KILL(not os.path.isfile(vocabPath), f"Vocabulary not found: {vocabPath}")
            KILL(not os.path.isfile(paramsPath), f"Parameters not found: {paramsPath}")

            with open(vocabPath, 'rb') as file:
                vocabs = pickle.load(file)
            toks_vocab = SeqVocabulary(vocabs[0])
            lbls_vocab = SeqVocabulary(vocabs[1])

            with open(paramsPath, 'rb') as file:
                params = json.load(file)

            self.toks_vocab = toks_vocab
            self.lbls_vocab = lbls_vocab


        self.word_embedding = nn.Embedding(params['vocab_size'], params['embedding_dim'])

        self.lstm = nn.LSTM(
            params['embedding_dim'], params['hidden_dim'],
            bidirectional=True,
            batch_first = True
        )

        lstm_output_dim = 2 * params['hidden_dim']
        self.dropout    = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(lstm_output_dim, params['num_classes'])

        
        # Set model for inference
        if isPretrained:
            self.load_state_dict(
                torch.load(modelPath, map_location=torch.device('cpu'))
            )
            self.eval()


    def classify(self, text, clean=True):
        '''
        text: (string) to be tokenized
        clean: (bool) whether to return the raw labels or divided by role.
        '''
        assert self.toks_vocab != None, "No pre-trained model has been loaded yet"

        seq = text.lower().split(" ")
        seq_idx = self.toks_vocab.preproc_data(seq).unsqueeze(0)
        out = self.forward(seq_idx.to(self.dev))
        out_idxs = torch.argmax(out, -1).tolist()[0]

        # Return raw classification
        if not clean:
            str_out = [self.lbls_vocab.vocab[x] for x in out_idxs]
        else:
            str_out = {}
            for i, x in enumerate(out_idxs):
                lbl = self.lbls_vocab.vocab[x]
                if lbl == 'O': continue

                if not lbl in str_out: str_out[lbl] = [seq[i]]
                else: str_out[lbl].append(seq[i])

            str_out = {k: " ".join(v) for k, v in str_out.items()}

        return str_out

    def forward(self, x, xlen=None):
        if not xlen: xlen = [len(x) for x in x]
        x = self.word_embedding(x)
        x = self.dropout(x)

        # The packing mechanism of pytorch is a way to avoid
        # useless computations for padding.
        x = pack_padded_sequence(x, xlen, batch_first = True, enforce_sorted = False)
        out, (h, c) = self.lstm(x)
        out, _ = pad_packed_sequence(out, batch_first = True)

        out = self.dropout(out)
        output = self.classifier(out)

        return output


def KILL(cond, message):
    if not cond: return
    print(message)
    exit(1)

