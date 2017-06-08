#!/usr/bin/python

import sys
from helpers import build_vocab

id_string = sys.argv[1]
ids = [int(item) for item in id_string.split()]
word2index, index2word = build_vocab('vocab.txt')

sentence = []
for idx in ids:
	sentence.append(index2word[idx])

print ' '.join(sentence)