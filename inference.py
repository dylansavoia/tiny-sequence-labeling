from pprint import pprint
from SeqModel import SeqVocabulary, SeqModel

import argparse
parser = argparse.ArgumentParser(description="Use a pretrained model over text")
parser.add_argument("--path", type=str, help='Path to pretrained model',
                    default="models/newestModel")
parser.add_argument("--sentence", type=str, help='Path to pretrained model',
                    default="Turn on the radio")
parser.add_argument("--raw", help='Return model outputs with no post-process',
                    action="store_true")
args = parser.parse_args()

sentence  = args.sentence
model = SeqModel(load=args.path)
out = model.classify(sentence, clean=not(args.raw))

print(sentence)
pprint(out)
