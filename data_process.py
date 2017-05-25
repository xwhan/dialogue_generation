#!/usr/bin/env python
import numpy as np
import gensim

all_data_path = 'OpenSubData/data.txt'

f = open(all_data_path)
data = f.readlines()
f.close()

datasize = len(data)
input_max = 0
output_max = 0

for sample in data:
	sample = sample[:-1]
	input_ = sample.split('|')[0]
	input_tokens = input_.split()
	if len(input_tokens) > input_max:
		input_max = len(input_tokens)
	output_ = sample.split('|')[1]
	output_tokens = output_.split()
	if len(output_tokens) > output_max:
		output_max = len(output_tokens)

print input_max
print output_max

