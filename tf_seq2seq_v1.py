#!/usr/bin/python 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf 
import numpy as np
import math

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell
import tensorflow.contrib.seq2seq as seq2seq

from helpers import *
from classifier import Classifier

class Seq2SeqAttn(object):
	"""seq2seq attention model"""
	def __init__(self, vocab_size, embed_size, num_layers, hidden_size, eos, max_len, initial_embed=None):
		super(Seq2SeqAttn, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.hidden_size = hidden_size

		# TO DO
		self.EOS = eos
		self.PAD = 0
		
		# placeholders
		self.encoder_inputs = tf.placeholder(shape=(None,None), dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
		self.rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name='rewards')
		# self.decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')

		# LSTM cell
		cells = []
		for _ in range(num_layers):
			cells.append(tf.contrib.rnn.LSTMCell(hidden_size))

		self.encoder_cell = MultiRNNCell(cells)
		self.decoder_cell = MultiRNNCell(cells)

		# decoder train feeds
		with tf.name_scope('decoder_feeds'):
			sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))
			EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
			PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD
			self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
			
			# self.decoder_train_length = self.decoder_targets_length + 1 # one for EOS
			self.decoder_train_length = tf.cast(tf.ones(shape=(batch_size,)) * (max_len+1),tf.int32)

			decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0) # (seq_len + 1) * batch_size
			# decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
			# decoder_train_targets_eos_mask = tf.one_hot(
				# self.decoder_targets_length,
				# decoder_train_targets_seq_len,
				# on_value=self.EOS,
				# off_value=self.PAD,
				# dtype=tf.int32
				# ) # batch_size * (seq_len + 1)
			# decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
			# decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
			self.decoder_train_targets = decoder_train_targets # (seq_len + 1) * batch_size with EOS at the end of each sentence
			# loss_weights
			loss_weights = tf.cast(tf.cast(self.decoder_train_targets,tf.bool),tf.float32)
			self.loss_weights = tf.transpose(loss_weights, perm=[1,0])
			# self.loss_weights = tf.ones([batch_size, tf.reduce_max(self.decoder_train_length)], dtype=tf.float32, name="loss_weights")


		# embedding layer
		with tf.variable_scope('embedding'):
			# if initial_embed:
			self.embedding = tf.Variable(initial_embed, name='matrix', dtype=tf.float32)
			# else:
				# self.embedding = tf.Variable(tf.random_normal([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size), name='matrix', dtype=tf.float32)
			self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
			self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.decoder_train_inputs)

		# encoder
		with tf.variable_scope('encoder'):
			self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
				cell=self.encoder_cell, 
				inputs=self.encoder_inputs_embedded,
				sequence_length=self.encoder_inputs_length, 
				time_major=True, 
				dtype=tf.float32)

		# decoder
		with tf.variable_scope('decoder') as scope:
			def output_fn(outputs):
				return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
			attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2]) # batch_size * seq_len * hidden_size

			(attention_keys,attention_values,attention_score_fn,attention_construct_fn) = seq2seq.prepare_attention(
				attention_states=attention_states,
				attention_option="bahdanau",
				num_units=self.hidden_size,
			)

			decoder_fn_train = seq2seq.attention_decoder_fn_train(
			encoder_state=self.encoder_state,
			attention_keys=attention_keys,
			attention_values=attention_values,
			attention_score_fn=attention_score_fn,
			attention_construct_fn=attention_construct_fn,
			name='attention_decoder'
			)

			decoder_fn_inference = seq2seq.attention_decoder_fn_inference(
			output_fn=output_fn,
			encoder_state=self.encoder_state,
			attention_keys=attention_keys,
			attention_values=attention_values,
			attention_score_fn=attention_score_fn,
			attention_construct_fn=attention_construct_fn,
			embeddings=self.embedding,
			start_of_sequence_id=self.EOS,
			end_of_sequence_id=self.EOS,
			maximum_length= max_len + 5,
			num_decoder_symbols=self.vocab_size,
			)

			self.decoder_outputs_train, self.decoder_state_train, self.decoder_context_state_train = seq2seq.dynamic_rnn_decoder(
			cell=self.decoder_cell,
			decoder_fn=decoder_fn_train,
			inputs=self.decoder_train_inputs_embedded,
			sequence_length=self.decoder_train_length,
			time_major=True,
			scope=scope,
			)

			self.decoder_logits_train = output_fn(self.decoder_outputs_train)
			self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

			scope.reuse_variables()

			self.decoder_logits_inference, self.decoder_state_inference, self.decoder_context_state_inference = (seq2seq.dynamic_rnn_decoder(
					cell=self.decoder_cell,
					decoder_fn=decoder_fn_inference,
					time_major=True,
					scope=scope,
				)
			)

			self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

		# optimizer
		with tf.name_scope('optimizer'):
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			logits = tf.transpose(self.decoder_logits_train, [1, 0, 2]) # batch_size * sequence_length * vocab_size
 			targets = tf.transpose(self.decoder_train_targets, [1, 0])
			output_prob = tf.reduce_max(tf.nn.softmax(logits), axis=2) # batch_size * seq_len
			seq_log_prob = tf.reduce_sum(tf.log(output_prob), axis=1)
			self.policy_loss = - tf.reduce_sum(self.rewards * seq_log_prob)
			self.policy_op = tf.train.AdamOptimizer().minimize(self.policy_loss)

 
			self.loss = seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights)
			self.train_op = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

	def policy_learning(self, input_seqs, input_lengths, rewards, sess=None):
		sess = sess or self.get_default_session()
		_, loss = sess.run([self.policy_op, self.policy_loss], {self.encoder_inputs:input_seqs, self.encoder_inputs_length:input_lengths, self.rewards:rewards})

		return loss

	def update(self, input_seqs, input_lengths, target_seqs, sess=None):
		sess = sess or tf.get_default_session()
		_, step, loss = sess.run([self.train_op, self.global_step, self.loss],{self.encoder_inputs:input_seqs, self.encoder_inputs_length:input_lengths, self.decoder_targets:target_seqs})

		return [step, loss]

def train():
	word2index, index2word = build_vocab('vocab.txt')
	print '----------PREPARING TRAINING DATA---------------'
	input_seqs, target_seqs = build_data('OpenSubData/first_100000.txt', word2index)
	matrix = np.load(open('embedding'))

	tf.reset_default_graph()
	generator = Seq2SeqAttn(25000 + 2, 100, 2, 128, 1, max_len=31, initial_embed=matrix)

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		num_epoch = 100
		eval_step = 100
		for epoch in range(num_epoch):
			batches = batch_generator(input_seqs, target_seqs, 256)
			for batch in batches:
				step, batch_loss = generator.update(batch[0], batch[1], batch[2])
				if step % eval_step == 0:
					print 'EPOCH: %d BATCH LOSS: %f UPDATE STEP: %d' % (epoch, batch_loss, step)
		saver.save(sess, 'models/attn_pretrained_' + str(num_epoch))
		print 'MODEL SAVED'

def rl_train():
	matrix = np.load(open('embedding'))
	# tf.reset_default_graph()
	# with tf.variable_scope('generator'):
	g1 = tf.Graph()
	with g1.as_default():
		generator = Seq2SeqAttn(25000 + 2, 100, 2, 128, 1, max_len=31, initial_embed=matrix)
		saver_1 = tf.train.Saver()
	# with tf.variable_scope('oracle'):

	g2 = tf.Graph()
	with g2.as_default():
		oracle = Classifier([3,4,5], 128, 30, 2, 25000, 100, l2_reg_lambda=0.1)
		saver_2 = tf.train.Saver()

	sess1 = tf.Session(graph=g1)
	saver_1.restore(sess1, 'models/attn_pretrained_100')
	sess2 = tf.Session(graph=g2)
	saver_2.restore(sess2, 'models/classifer')

if __name__ == '__main__':
	# rl_train()
	train()





