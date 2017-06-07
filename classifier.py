#!/usr/bin/env python

import tensorflow as tf
import numpy as np 

class Classifier(object):
	"""CNN classifier"""
	def __init__(self, filter_sizes, num_filters, seq_len, num_classes, vocab_size, embed_size, l2_reg_lambda=0.0):
		super(Classifier, self).__init__()
		self.filter_sizes = filter_sizes
		self.num_filters = num_filters
		self.l2_reg_lambda = l2_reg_lambda
		self.seq_len = seq_len

		# Placeholders
		self.input_x = tf.placeholder(tf.int32, [None, seq_len], name='input')
		self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='label')
		self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

		l2_loss = tf.constant(0.0)

		with tf.variable_scope('embedding'):
			self.W = tf.Variable(tf.random_uniform([vocab_size, embed_size], - 0.5 / embed_size, 0.5 / embed_size), name='embed')
			self.embed_chars = tf.nn.embedding_lookup(self.W, self.input_x)
			self.embed_chars_expanded = tf.expand_dims(self.embed_chars, -1)

		pooling = []
		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				filter_shape = [filter_size, embed_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
				b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
				conv = tf.nn.conv2d(
					self.embed_chars_expanded,
					W,
					strides=[1, 1, 1, 1],
					padding="VALID",
					name="conv")
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, seq_len - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="pool")
				pooling.append(pooled)

		num_filters_total = num_filters * len(filter_sizes)
		self.h_pool = tf.concat(pooling, 3)
		self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

		with tf.name_scope("dropout"):
			self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)


		with tf.variable_scope('output'):
			W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
			b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
			l2_loss += tf.nn.l2_loss(W)
			l2_loss += tf.nn.l2_loss(b)
			self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
			self.predictions = tf.argmax(self.scores, 1, name="predictions")

		with tf.variable_scope('loss'):
			losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
			self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

		with tf.variable_scope('optimize'):
			self.global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-4)
			grads_and_vars = optimizer.compute_gradients(self.loss)
			self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

		with tf.variable_scope('accuracy'):
			correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

	def update_step(self, data_x, data_y, dropout, sess=None):
		sess = sess or tf.get_default_session()
		_, step, loss, accuracy = sess.run([self.train_op,self.global_step, self.loss, self.accuracy], {self.input_x:data_x, self.input_y:data_y,self.dropout_keep_prob:dropout})
		return [step, loss, accuracy]


	def predict(self,data_x, data_y, sess=None):
		sess = sess or tf.get_default_session()
		loss, accuracy, predictions = sess.run([self.loss, self.accuracy, self.predictions],{self.input_x:data_x, self.input_y:data_y, self.dropout_keep_prob:1.0})
		return [loss, accuracy, predictions]

	def inference(self, data_x, sess=None):
		sess = sess or tf.get_default_session()
		predictions = sess.run(self.predictions, {self.input_x:data_x, self.dropout_keep_prob:1.0})
		return predictions

if __name__ == '__main__':
	a = Classifier([3,4,5], 128, 30, 2, 25000, 100)



