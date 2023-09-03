from .seqVocabulary import SeqVocabulary
from torch.utils.data import IterableDataset
from itertools import chain
import pickle


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



