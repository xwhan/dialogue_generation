#!/usr/bin/python 

import tensorflow as tf 
import numpy as np

from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

class Seq2SeqAttn(object):
	"""seq2seq attention model"""
	def __init__(self, vocab_size, embed_size, num_layers, hidden_size):
		super(Seq2SeqAttn, self).__init__()
		self.vocab_size = vocab_size
		self.embed_size = embed_size
		self.num_layers = num_layers
		self.hidden_size = hidden_size
		
		self.encoder_input = tf.placeholder(shape=(None,None), dtype=tf.int32, name='encoder_inputs')
		self.encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')

		self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

		# Embedding Layer Can be replaced with pre-trained embeddings later
		self.embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
		self.encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

		encoder_cell = LSTMCell(encoder_hidden_units)