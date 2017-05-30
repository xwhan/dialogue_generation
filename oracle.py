#!/usr/bin/python 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf
from classifier import Classifier

from tqdm import tqdm

emotional_ends = {14, 3823, 6755}

def get_data(datapath, max_len):
	data = open(datapath).readlines()
	data_size = len(data)
	seqs = []
	labels = []
	for line in data:
		pairs = line.rstrip().split('|')
		seq = []
		for word in pairs[1].split():
			seq.append(int(word))
		if seq[-1] in emotional_ends:
			labels.append(1)
			seq = seq[:-1]
		else:
			labels.append(0)
		seqs.append(seq)	
	input_matrix = np.zeros([data_size, max_len],dtype=np.int32)
	for i, seq in enumerate(seqs):
		for j, elem in enumerate(seq):
			input_matrix[i,j] = elem
	labels_onehot = np.zeros((len(labels),2))
	labels_onehot[np.arange(len(labels)), np.array(labels)] = 1	
	return [input_matrix, labels_onehot, data_size]

def build_data(input_matrix, labels_onehot, data_size, batch_size):
	print "PREPARING DATA"
	num_batches = int((data_size - 1)/batch_size) + 1
	shuffle_indices = np.random.permutation(np.arange(data_size))
	input_shuffle = input_matrix[shuffle_indices]
	label_shuffle = labels_onehot[shuffle_indices]

	for batch in range(num_batches):
		start_index = batch * batch_size
		end_index = min((batch + 1) * batch_size, data_size)
		yield [input_shuffle[start_index:end_index], label_shuffle[start_index:end_index]]

if __name__ == '__main__':
	tf.reset_default_graph()
	max_len = 30
	classifier = Classifier([3,4,5], 128, max_len, 2, 25000, 100, l2_reg_lambda=0.1)
	saver = tf.train.Saver()

	print 'LOADING DATA'
	train_data, train_labels, train_size = get_data('OpenSubData/xaa', max_len)
	test_data, test_label, _ = get_data('OpenSubData/mini_data.txt', max_len)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		num_epoch = 10
		for epoch in range(num_epoch):
			batches = build_data(train_data, train_labels, train_size, 128)
			print 'START TRAINING EPOCH:%d' % epoch 
			for batch in batches:
				step, loss, accuracy = classifier.update_step(batch[0], batch[1], 0.5)
				if step % 5000 == 0:
					print 'BATCH LOSS AFTER %d step: %f' % (step, loss) 
			curr_loss, curr_accuracy, predictions = classifier.predict(test_data, test_label)
			print '\nVALIDATION ERROR & ACCURACY AFTER %d EPOCH: %f, %f' % (epoch+1, curr_loss, curr_accuracy)

		saver.save(sess, 'models/classifer')
		print 'MODEL SAVED'




