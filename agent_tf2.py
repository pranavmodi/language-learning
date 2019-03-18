import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class LinearSigmoid(Layer):

  def __init__(self, input_dim=32, h_units=32):
      super(LinearSigmoid, self).__init__()
      w_init = tf.random_normal_initializer(stddev=0.01)
      self.w = tf.Variable(
          initial_value=w_init(shape=(input_dim, h_units), dtype='float32'),
          trainable=True)
      b_init = tf.zeros_initializer()
      self.b = tf.Variable(
          initial_value=b_init(shape=(h_units,), dtype='float32'),
          trainable=True)

  def call(self, inputs):
      return tf.sigmoid(tf.matmul(inputs, self.w) + self.b)


class Agents:
    """ This class contains the sender network which recieves the activations 
        of the target image and distractor image, maps them into vocabulary words
        and sends it the reciever,  and also the reciever network which gets the 
        word, maps it into it's embedding and selects an image to recieve the feedback.
    """

    def __init__(self, vocab, image_embedding_dim, word_embedding_dim,
                 learning_rate, temperature=10, batch_size=32):

        self.vocab = vocab
        self.image_embedding_dim = image_embedding_dim
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.sender_embed = LinearSigmoid(1000, 10)
        self.receiver_embed = LinearSigmoid(1000, 10)

        w_init = tf.random_normal_initializer(stddev=0.01)
        gsi_shape = ((2 * self.image_embedding_dim), len(self.vocab))
        self.gsi_embed = tf.Variable(initial_value=w_init(gsi_shape), dtype='float32', trainable=True)
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)
        vocab_scores = tf.matmul(ordered_embed, self.gsi_embed)
        self.word_probs = tf.nn.softmax(vocab_scores).numpy()[0]
        #self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

    def sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)


    def get_sender_word_probs(self, target_acts, distractor_acts):
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)
        self.word_probs_model(ordered_embed)


    def sender_word_model(self):
        self.word_probs_model = tf.keras.Sequential()
        self.word_probs_model.add(layers.Dense(2, use_bias=True, input_shape=(2 * self.image_embedding_dim),
                                               activation='softmax'))


if __name__=='__main__':

    vocab = ['dog', 'cat', 'mouse']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10,
                   learning_rate=0.2, temperature=10, batch_size=32)
    t_acts = tf.ones((1, 1000))
    d_acts = tf.ones((1, 1000))

    word_probs = agent.get_sender_word(t_acts, d_acts)


        
