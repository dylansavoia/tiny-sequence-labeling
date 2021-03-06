import torch
from torch import nn
from itertools import chain
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import pickle
import json
import io


NAME = ''    # used when saving model, vocabulary and parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print("Using {}".format(DEVICE))

MODELSPATH = 'models/model.pth' # Where to save trained model
DATAPATH   = 'data/dataset.txt'
PARAMS = {
    "hidden_dim": 30,
    "embedding_dim": 30,
    "dropout": 0.1,
}


def main ():
    trainer = Trainer(
        modelpath=MODELSPATH, datapath=DATAPATH,
        BS=512,
        dev=DEVICE,
        params=PARAMS
    )
    trainer.train(10)

class Trainer():
    def __init__(self,
            modelpath='', datapath='',
            BS=32, dev=torch.device('cpu'),
            params=None
        ):
        # Load dataset
        self.dataset = SeqDataset(modelpath, datapath)
        self.modelpath = modelpath
        self.trainParams = params

        self.dataloader = DataLoader(self.dataset, BS, collate_fn=pad_collate)
        self.trainParams['vocab_size'] = len(self.dataset.toks_vocab)
        self.trainParams['num_classes'] = len(self.dataset.lbls_vocab)

        # Save params to be used in inference.py
        params_file_name = modelpath.rsplit('.', 1)[0] + ".json"
        with open(params_file_name, 'w') as f:
            json.dump(self.trainParams, f)

        self.model = SeqModel(self.trainParams, dev=dev);
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.model.train()
        self.model.to(dev)
        self.dev = dev

        # <PAD> is placed as the first element of the vocabulary
        # exactly for this purpose: see SeqDataset
        self.loss_function = nn.CrossEntropyLoss(ignore_index=0)


    def train(self, epochs):
        print('Training:')

        for epoch in range(epochs):
            epoch_loss = 0.0
            nsamples = 0
            for i, data in tqdm(enumerate(self.dataloader)):
                x, y, xlen = data
                x = x.to(self.dev)

                out  = self.model(x, xlen)
                out  = out.view(-1, out.shape[-1])
                y    = y.view(-1).to(self.dev)
                loss = self.loss_function(out, y)

                epoch_loss += loss.tolist()
                nsamples   += x.size(0)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Tot. Epoch Loss: {epoch_loss}")

        torch.save(self.model.state_dict(), self.modelpath)
        print("Checkpoint Saved")


class SeqVocabulary:
    def __init__(self, vocab):
        assert len(vocab) > 0
        self.vocab   = vocab
        self.unk_idx = vocab.index("<UNK>")
        self.str2idx_dict = {k:i for i,k in enumerate(self.vocab)}

    def str2idx (self, w):
        if not w in self.vocab:
            return self.unk_idx
        else: return self.str2idx_dict[w]

    def preproc_data(self, data):
        lidx = [self.str2idx(x) for x in data]
        lidx = torch.LongTensor(lidx)
        return lidx

    def __len__(self):
        return len(self.vocab)


class SeqDataset(IterableDataset):
    def __init__(self, modelpath, datapath):
        '''
        Generate vocabulary for model.
        If datapath is not specified,
        a vocabulary is assumed to exist.
        '''

        vocab_file_name = modelpath.rsplit('.', 1)[0] + ".pkl"
        if datapath:
            self.toks, self.lbls = SeqDataset.read_dataset(datapath)
            self.toks_vocab = ['<UNK>'] + list(set(chain(*self.toks)))
            self.lbls_vocab = ['<PAD>', '<UNK>'] + list(set(chain(*self.lbls)))

            with open(vocab_file_name, 'wb') as f:
                pickle.dump(
                    [self.toks_vocab, self.lbls_vocab],
                    f
                )

        else:
            with open(vocab_file_name, 'rb') as f:
                vocabs = pickle.load(f)
            self.toks_vocab = vocabs[0]
            self.lbls_vocab = vocabs[1]

        self.toks_vocab = SeqVocabulary(self.toks_vocab)
        self.lbls_vocab = SeqVocabulary(self.lbls_vocab)

    def __iter__(self):
        for l, t in zip(self.toks, self.lbls):
            lidx = self.toks_vocab.preproc_data(l)
            tidx = self.lbls_vocab.preproc_data(t)
            yield (lidx, tidx)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_dataset(path):
        f = open(path, 'r')
        line = True
        all_toks, all_lbls = [], []

        while line:
            line = f.readline().lower()
            cline = line.rstrip()
            if len(cline) < 1: continue

            lbls = f.readline().rstrip()
            cline = cline.split(' ')
            lbls  = lbls.split(' ')

            assert len(cline) == len(lbls)
            all_toks.append(cline)
            all_lbls.append(lbls)

        f.close()
        return all_toks, all_lbls


class SeqModel(nn.Module):
    def __init__(self, params, dev=torch.device('cpu')):
        super(SeqModel, self).__init__()
        self.dev = dev

        self.word_embedding = nn.Embedding(params['vocab_size'], params['embedding_dim'])

        self.lstm = nn.LSTM(
            params['embedding_dim'], params['hidden_dim'],
            bidirectional=True,
            batch_first = True
        )

        lstm_output_dim = 2 * params['hidden_dim']
        self.dropout = nn.Dropout(params['dropout'])
        self.classifier = nn.Linear(lstm_output_dim, params['num_classes'])

    def classify(self, text, toks_vocab, lbls_vocab):
        seq = text.lower().split(" ")
        seq_idx = toks_vocab.preproc_data(seq).unsqueeze(0)
        out = self.forward(seq_idx.to(self.dev))
        out_idxs = torch.argmax(out, -1).tolist()[0]

        str_out = [lbls_vocab.vocab[x] for x in out_idxs]
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

# Utils
def pad_collate(batch):
    x, y = zip(*batch)
    xlens = [len(x) for x in x]
    x = pad_sequence(x, batch_first = True)
    y = pad_sequence(y, batch_first = True)
    return (x, y, xlens)

if __name__ == '__main__':
    main()
