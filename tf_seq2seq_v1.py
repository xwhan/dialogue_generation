#!/usr/bin/python 

import tensorflow as tf 
import numpy as np
import math

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple, MultiRNNCell
import tensorflow.contrib.seq2seq as seq2seq

import helpers

class Seq2SeqAttn(object):
	"""seq2seq attention model"""
	def __init__(self, vocab_size, embed_size, num_layers, hidden_size, eos, initial_embed=None):
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
		self.decoder_targets_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='decoder_targets_length')

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
			self.decoder_train_length = self.decoder_targets_length + 1 # one for EOS

			decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0) # (seq_len + 1) * batch_size
			decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
			decoder_train_targets_eos_mask = tf.one_hot(
				self.decoder_targets_length,
				decoder_train_targets_seq_len,
				on_value=self.EOS,
				off_value=self.PAD,
				dtype=tf.int32
				) # batch_size * (seq_len + 1)
			decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
			decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)
			self.decoder_train_targets = decoder_train_targets # (seq_len + 1) * batch_size with EOS at the end of each sentence
			# loss_weights
			# loss_weights = tf.cast(tf.cast(self.decoder_targets,tf.bool),tf.float32)
			# self.loss_weights = tf.transpose(loss_weights, perm=[1,0])
			self.loss_weights = tf.ones([batch_size, tf.reduce_max(self.decoder_train_length)], dtype=tf.float32, name="loss_weights")


		# embedding layer
		with tf.variable_scope('embedding'):
			if initial_embed:
				self.embedding = tf.Variable(initial_embed, name='matrix', dtype=tf.float32)
			else:
				self.embedding = tf.Variable(tf.random_normal([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size), name='matrix', dtype=tf.float32)
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
			attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

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
			maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
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
			logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
			targets = tf.transpose(self.decoder_train_targets, [1, 0])
			self.loss = seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights)
			self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

	def make_train_inputs(self, input_seq, target_seq):
	 	inputs_, inputs_length_ = helpers.batch(input_seq)
		targets_, targets_length_ = helpers.batch(target_seq)
		return {
			self.encoder_inputs: inputs_,
			self.encoder_inputs_length: inputs_length_,
			self.decoder_targets: targets_,
			self.decoder_targets_length: targets_length_,
		}

	def make_inference_inputs(self, input_seq):
		inputs_, inputs_length_ = helpers.batch(input_seq)
		return {
		self.encoder_inputs: inputs_,
		self.encoder_inputs_length: inputs_length_,
		}



if __name__ == '__main__':

	generator = Seq2SeqAttn(200, 100, 2, 128, 1)








