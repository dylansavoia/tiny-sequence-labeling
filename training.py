import torch
import json

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from pprint import pprint
from tqdm import tqdm

from SeqModel import SeqDataset, SeqModel

import argparse
parser = argparse.ArgumentParser(description="Train a new Sequence model.")
parser.add_argument("--path", type=str, help='Model save path',
                    default="models/newestModel")
parser.add_argument("--data", type=str, help="Path to dataset (plaintext)",
                    default="data/dataset.txt")
parser.add_argument("--lr", type=float, help="Learning rate",
                    default=0.01)
parser.add_argument("--nepochs", type=int, help="Number of epochs",
                    default=10)
parser.add_argument("--hidden_dim", type=int, help="Size of hidden vector",
                    default=30)
parser.add_argument("--embedding_dim", type=int, help="Size of each embedding vector",
                    default=30)
parser.add_argument("--dropout", type=float, help="Dropout value for regularization",
                    default=0.1)

args = parser.parse_args()

LEARNING_RATE = args.lr
N_EPOCHS      = args.nepochs

MODELSPATH = args.path # Where to save trained model
DATAPATH   = args.data # Path to dataset

PARAMS = {
    "hidden_dim": args.hidden_dim,
    "embedding_dim": args.embedding_dim,
    "dropout": args.dropout
}


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
print("Using {}".format(DEVICE))
pprint(vars(args))

def main ():
    trainer = Trainer(
        modelpath=MODELSPATH, datapath=DATAPATH,
        BS=512,
        dev=DEVICE,
        params=PARAMS
    )
    trainer.train(N_EPOCHS)

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
        with open(f"{modelpath}.json", 'w') as f:
            json.dump(self.trainParams, f)

        self.model = SeqModel(self.trainParams, dev=dev);
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        self.model.train()
        self.model.to(dev)
        self.dev = dev

        # <PAD> is placed as the first element of the vocabulary
        # exactly for this purpose: see SeqDataset
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)


    def train(self, epochs):
        print('\nTraining:')

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

        torch.save(self.model.state_dict(), f"{self.modelpath}.pth")
        print("Checkpoint Saved")


# Utils
def pad_collate(batch):
    x, y = zip(*batch)
    xlens = [len(x) for x in x]
    x = pad_sequence(x, batch_first = True)
    y = pad_sequence(y, batch_first = True)
    return (x, y, xlens)

if __name__ == '__main__':
    main()
