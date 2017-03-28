import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
import utils
import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class SenderAgent:

    def __init__(self, vocab, embedding_dim = 2):
        self.vocab = vocab
        self.embedding_dim = embedding_dim

        self._build_image_to_vocab_mapper()

    def _build_image_to_vocab_mapper(self):
        #with tf.variable_scope('image_to_word'):
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. sigmoid non-linearity on embedding space
        # 3. fc-weights to concatenated vector for scores over vocacb
        with tf.name_scope('sender'):
            self.im_activations = tf.placeholder("float", [2, 1000])
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            embed_im = tf.nn.sigmoid(tf.matmul(self.im_activations, weights1))
            concat_im = tf.reshape(embed_im, [1, 2*self.embedding_dim])

            weights2 = tf.random_normal([2*self.embedding_dim, len(self.vocab)], stddev=0.1)
            self.vocab_scores = tf.matmul(concat_im, weights2)


    def show_images(self, sess, target_acts, distractor_acts):
        batch = np.concatenate([[target_acts, distractor_acts]], axis=0)
        v_scores = sess.run(self.vocab_scores, feed_dict={self.im_activations : batch})
        print(v_scores)
        comm_word = [0,1]
        return comm_word



class RecieverAgent:
    def __init__(self, vocab, activation_shape=[1000, 1]):
        self.something = []
        self.vocab = vocab

        self.vgg = vgg16.Vgg16()

    def show_images(self, sess, comm_word, image1, image2):
        print('got comm word: ', comm_word)
        return 0


    def _build_image_selection_with_word():
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. Symbol to embedding space
        # 3. Dot product of symbol and each image

        with tf.name_scope('reciever'):
            im_activations = tf.placeholder("float", [2, 1000])
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            embed_im = tf.nn.sigmoid(tf.matmul(im_activations, weights1))
            concat_im = tf.reshape(embed_im, [1, 2*embedding_dim])
            weights2 = tf.random_normal([2*embedding_dim, len(vocab)], stddev=0.1)
            self.vocab_scores = tf.matmul(concat_im, weights2)
