#!/usr/bin/python

import os
import numpy as np 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf 
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell


from helpers import *

class Seq2SeqAttn(object):
	"""loop flow version of seq2seq model"""
	def __init__(self, vocab_size, embed_size, hidden_size, num_layers, eos, max_len=31, initial_embed=None):
		super(Seq2SeqAttn, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.hidden_size = hidden_size

		self.EOS = eos
		self.PAD = 0

		## Data Placeholders
		self.encoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

		seq_len, batch_size = tf.unstack(tf.shape(self.decoder_targets))
		loss_weights = tf.cast(tf.cast(self.decoder_targets,tf.bool),tf.float32)
		self.loss_weights = tf.transpose(loss_weights, perm=[1,0])
		# self.loss_weights = tf.ones(shape=(batch_size, seq_len))

		## Embedding Layer		
		with tf.variable_scope('embedding'):
			#if initial_embed:
			self.embedding = tf.Variable(initial_embed, name='matrix', dtype=tf.float32, trainable=True)
			# else:
				# self.embedding = tf.Variable(tf.random_normal([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size), name='matrix', dtype=tf.float32)
			self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)

		## Encoder & Decoder Cells
		cells = []
		for _ in range(num_layers):
			cells.append(LSTMCell(hidden_size))
		self.encoder_cell = MultiRNNCell(cells)
		self.decoder_cell = MultiRNNCell(cells)

		self.encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(self.encoder_cell, inputs=self.encoder_inputs_embedded, sequence_length=self.encoder_inputs_length, dtype=tf.float32, time_major=True)

		self.encoder_max_time, self.batch_size = tf.unstack(tf.shape(self.encoder_inputs))

		self.decoder_lengths = tf.cast(tf.ones(shape=(self.batch_size,)) * max_len,tf.int32)

		## Output projection
		with tf.variable_scope('output'):
			self.W = tf.Variable(tf.random_uniform([hidden_size, vocab_size], -1, 1), dtype=tf.float32)
			self.b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

		eos_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='EOS') * self.EOS
		pad_time_slice = tf.ones([self.batch_size], dtype=tf.int32, name='PAD') * self.PAD

		self.eos_step_embedded = tf.nn.embedding_lookup(self.embedding, eos_time_slice)
		self.pad_step_embedded = tf.nn.embedding_lookup(self.embedding, pad_time_slice)

		def loop_fn_initial():
			initial_elements_finished = (0 >= self.decoder_lengths)
			initial_input = self.eos_step_embedded
			initial_cell_state = self.encoder_final_state
			initial_cell_output = None
			initial_loop_state = None
			return (initial_elements_finished,
				initial_input,
				initial_cell_state,
				initial_cell_output,
				initial_loop_state)

		def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
			# TO DO, add randomness about whether feed previous or teacher forcing
			def get_next_input():
				output_logits = tf.add(tf.matmul(previous_output, self.W), self.b)
				prediction = tf.argmax(output_logits, axis=1)
				next_input = tf.nn.embedding_lookup(self.embedding, prediction)
				return next_input

			elements_finished = (time >= self.decoder_lengths)
			finished = tf.reduce_all(elements_finished)
			input = tf.cond(finished, lambda: self.pad_step_embedded, get_next_input)
			state = previous_state
			output = previous_output
			loop_state = None

			return (elements_finished, 
					input,
					state,
					output,
					loop_state)

		def loop_fn(time, previous_output, previous_state, previous_loop_state):
			if previous_state is None:    # time == 0
				assert previous_output is None and previous_state is None
				return loop_fn_initial()
			else:
				return loop_fn_transition(time, previous_output,previous_state, previous_loop_state)

		with tf.variable_scope('decoder'):
			self.decoder_outputs_ta, self.decoder_final_state, _ = tf.nn.raw_rnn(self.decoder_cell, loop_fn)

		self.decoder_outputs = self.decoder_outputs_ta.stack()

		decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(self.decoder_outputs))
		decoder_outputs_flat = tf.reshape(self.decoder_outputs, (-1, decoder_dim))
		decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, self.W), self.b)
		decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, self.vocab_size))
		decoder_logits = tf.transpose(decoder_logits, perm=[1,0,2]) # batch_size * max_steps * vocab_size
		decoder_targets = tf.transpose(self.decoder_targets, perm=[1,0])

		self.decoder_prediction = tf.argmax(decoder_logits, 2)

		## Optimizer & Loss
		# stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)
		self.global_step = tf.Variable(0, name="global_step", trainable=False)
		# self.loss = tf.reduce_mean(stepwise_cross_entropy)
		self.loss = seq2seq.sequence_loss(logits=decoder_logits, targets=decoder_targets, weights=self.loss_weights)
		self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

	def update(self, input_data, target_data, input_lengths, sess=None):
		sess = sess or tf.get_default_session()
		_, step, loss = sess.run([self.train_op, self.global_step, self.loss],{self.encoder_inputs:input_data, self.encoder_inputs_length:input_lengths, self.decoder_targets:target_data})
		return [step, loss]

	def generate(self, input_data, input_lengths, sess=None):
		sess = sess or tf.get_default_session()
		prediction = sess.run(self.decoder_prediction, {self.encoder_inputs:input_data, self.encoder_inputs_length:input_lengths})
		return prediction

def train():
	word2index, index2word = build_vocab('vocab.txt')
	
	# get train data
	print '----------PREPARING TRAINING DATA---------------'
	input_seqs, target_seqs = build_data('OpenSubData/first_10000.txt', word2index)
	#matrix = embed_fromGlove('glove.twitter.27B.100d.txt', index2word)
	matrix = np.load(open('embedding'))

	tf.reset_default_graph()
	generator = Seq2SeqAttn(25000 + 2, 100, 128, 2, word2index['EOS'], initial_embed=matrix)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		num_epoch = 500
		eval_step = 100
		for epoch in range(num_epoch):
			# print "EPOCH %d" % epoch
			batches = batch_generator(input_seqs, target_seqs, 128)
			for batch in batches:
				step, batch_loss = generator.update(batch[0], batch[2], batch[1])
				if step % eval_step == 0:
					print 'EPOCH: %d BATCH LOSS: %f UPDATE STEP: %d' % (epoch, batch_loss, step)
		saver.save(sess, 'models/seq2seq_pretrained_' + str(num_epoch))
		print 'MODEL SAVED'

def test():
	word2index, index2word = build_vocab('vocab.txt')
	print '----------PREPARING TESTING DATA---------------'
	input_seqs, target_seqs = build_data('OpenSubData/for_generation.txt', word2index)

	matrix = np.load(open('embedding'))
	tf.reset_default_graph()
	generator = Seq2SeqAttn(25000 + 2, 100, 128, 2, word2index['EOS'], initial_embed=matrix)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		saver.restore(sess, 'models/seq2seq_pretrained_100')
		print 'MODEL RESTORED'
		batch_size = 5
		batches = batch_generator(input_seqs, target_seqs, batch_size)
		for batch in batches:
			inputs = batch[0].T
			# targets = batch[2].T
			outputs = generator.generate(batch[0], batch[1], sess)
			# print inputs, targets, outputs
			# break
			for idx in range(batch_size):
				input_seq = inputs[idx,:]
				output_seq = outputs[idx,:]
				input_sent = []
				output_sent = []
				for num in input_seq:
					if num == 0:
						break
					else:
						input_sent.append(index2word[num])
				for num in output_seq:
					if num == word2index['EOS']:
						break
					else:
						output_sent.append(index2word[num])
				print 'INPUT: ', ' '.join(input_sent)
				print 'OUTPUT: ', ' '.join(output_sent)
			break

if __name__ == '__main__':
	train()
	# test()

