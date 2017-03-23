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

        self.vgg = vgg16.Vgg16()
        self.images = tf.placeholder("float", [2, 224, 224, 3])
        self.vgg.build(self.images)
        self.vocab = vocab
        self.embedding_dim = embedding_dim

    def _build_image_to_vocab_mapper():
        #with tf.variable_scope('image_to_word'):
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. sigmoid non-linearity on embedding space
        # 3. fc-weights to concatenated vector for scores over vocacb
        im_activations = tf.placeholder("float", [2, 1000])
        weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
        embed_im = tf.nn.sigmoid(tf.matmul(im_activations, weights1))
        concat_im = tf.reshape(embed_im, [1, 2*embedding_dim])

        weights2 = tf.random_normal([2*embedding_dim, len(vocab)], stddev=0.1)
        self.vocab_scores = tf.matmul(concat_im, weights2)



    def show_images(self, sess, target, distractor):

        batch1 = target.reshape((1, 224, 224, 3))
        batch2 = distractor.reshape((1, 224, 224, 3))

        batch = np.concatenate((batch1, batch2), 0)
        #feed_dict = {images: batch}

        feed_dict = {self.images: batch}

        with tf.name_scope("content_vgg"):
            fc8 = sess.run(self.vgg.fc8, feed_dict=feed_dict)

        print(type(fc8))

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

        im_activations = tf.placeholder("float", [2, 1000])
        weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
        embed_im = tf.nn.sigmoid(tf.matmul(im_activations, weights1))
        concat_im = tf.reshape(embed_im, [1, 2*embedding_dim])

        weights2 = tf.random_normal([2*embedding_dim, len(vocab)], stddev=0.1)
        self.vocab_scores = tf.matmul(concat_im, weights2)
