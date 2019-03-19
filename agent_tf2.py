import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam


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
        self.vocab_len = len(self.vocab)
        self.build_sender_receiver_model()
        self.build_word_probs_model()
        self.sender_optimizer = Adam(self.learning_rate)
        self.receiver_optimizer = Adam(self.learning_rate)
        w_init = tf.random_normal_initializer(stddev=0.01)
        self.vocab_embedding = tf.Variable(w_init(shape=(self.vocab_len,
                                                         self.word_embedding_dim),
                                                  dtype='float32'),
                                           trainable=True)

        # self.sender_loss = tf.reduce_mean(-1 * tf.multiply(tf.transpose(tf.log(selected_word_prob)),
        #                                                    self.reward))

    def build_word_probs_model(self):
        self.word_probs_model = tf.keras.Sequential()
        self.word_probs_model.add(Dense(2, use_bias=True,
                                               input_shape=(None, 2*self.image_embedding_dim),
                                               activation='softmax'))


    def get_receiver_image_probs(self, word, im1_acts, im2_acts):
        word_embed = tf.squeeze(tf.gather(self.vocab_embedding, self.word))
        print(word_embed, 'word embed shape')
        im1_embed = self.receiver(im1_acts)
        im2_embed = self.receiver(im2_acts)
        print(im1_embed, 'im1_embed')
        print(im2_embed, 'im2_embed')
        print(tf.multiply(im1_embed, word_embed), 'im1_embed_mult')
        print(tf.multiply(im2_embed, word_embed), 'im2_embed_mult')
        im1_score = tf.reduce_sum(tf.multiply(im1_embed, word_embed), axis=1).numpy()
        im2_score = tf.reduce_sum(tf.multiply(im2_embed, word_embed), axis=1).numpy()
        print(im1_score, im2_score, im1_score == im2_score, 'the scores')
        image_probs = tf.nn.softmax([im1_score, im2_score]).numpy()
        return image_probs
      

    def build_sender_receiver_model(self):
        self.sender = LinearSigmoid(1000, self.image_embedding_dim)
        self.receiver = LinearSigmoid(1000, self.image_embedding_dim)


    def get_sender_word_probs(self, target_acts, distractor_acts):
        t_embed = self.sender(target_acts)
        d_embed = self.sender(distractor_acts)
        ordered_embed = tf.concat([t_embed, d_embed], axis=1)
        self.word_probs = self.word_probs_model(ordered_embed).numpy()[0]
        self.word = np.random.choice(np.arange(len(self.vocab)), p=self.word_probs)
        
        return self.word_probs, self.word


if __name__=='__main__':

    vocab = ['dog', 'cat', 'mouse']
    agent = Agents(vocab=vocab, image_embedding_dim=10, word_embedding_dim=10,
                   learning_rate=0.2, temperature=10, batch_size=32)
    t_acts = tf.ones((1, 1000))
    d_acts = tf.ones((1, 1000))

    word_probs = agent.get_sender_word(t_acts, d_acts)


        
