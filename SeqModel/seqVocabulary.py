import torch

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



