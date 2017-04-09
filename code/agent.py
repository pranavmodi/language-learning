import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
import utils
import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class Agents:

    def __init__(self, vocab, embedding_dim = 2):
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.im_activations = tf.placeholder("float", [2, 1000])
        self.target_acts = tf.placeholder("float", [None, 1000])
        self.distractor_acts = tf.placeholder("float", [None, 1000])
        self.vocab_scores = None
        self.game_scores = tf.placeholder(tf.float32, [None, 1])
        #self._build_image_to_vocab_mapper()
        self._trial_learning_graph()


    def _build_image_to_vocab_mapper(self):
        #with tf.variable_scope('image_to_word'):
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. sigmoid non-linearity on embedding space
        # 3. fc-weights to concatenated vector for scores over vocacb
        with tf.name_scope('sender'):

            target_weights = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            distractor_weights = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))

            #embed_im = tf.nn.sigmoid(tf.matmul(self.im_activations, weights1))
            #concat_im = tf.reshape(embed_im, [1, 2*self.embedding_dim])

            t_embed = tf.nn.sigmoid(tf.matmul(self.target_acts, target_weights))
            d_embed = tf.nn.sigmoid(tf.matmul(self.distractor_acts, distractor_weights))

            concat_im = tf.concat_v2([t_embed, d_embed], axis=1)

            weights2 = tf.Variable(tf.random_normal([2*(self.embedding_dim), self.embedding_dim], stddev=0.1))

            concat_embed = tf.matmul(concat_im, weights2)

            word_embed = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim], stddev=0.1))

            word_dot = tf.mul(concat_embed, word_embed)
            self.vocab_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)
            print(self.vocab_scores)
            self.loss = tf.reduce_sum(self.game_scores)
            print(self.loss)

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss)


    def _trial_learning_graph(self):


        with tf.name_scope('sender'):
            weights = tf.Variable(tf.random_normal([2000, self.embedding_dim], stddev=0.1))
            image_acts = tf.concat_v2([self.target_acts, self.distractor_acts], axis=1)
            self.vocab_scores = tf.sigmoid(tf.matmul(image_acts, weights))

            self.loss = tf.reduce_sum(self.game_scores)

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.loss)


    def show_images(self, sess, target_acts, distractor_acts):
        batch = np.concatenate([target_acts, distractor_acts], axis=1)
        v_scores = sess.run(self.vocab_scores, feed_dict={self.im_activations : batch})[0]
        comm_word = np.argmax(v_scores)
        return comm_word

    def train_sender(sess, samples, rewards):
        return 0


    def show_images_rec(self, sess, comm_word, image1, image2):
        batch = np.concatenate((image1, image2), axis=0)
        image_scores = sess.run(self.image_scores, feed_dict={self.word : comm_word, self.im_activations : batch})
        selected = np.argmax(image_scores)
        return selected


    def _build_image_selection_with_word(self):
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. Symbol to embedding space
        # 3. Dot product of symbol and each image

        with tf.name_scope('reciever'):
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim]))
            word_embed = vocab_embedding[self.word, :]
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            self.im_activations = tf.Print(self.im_activations, data = [self.im_activations.get_shape()], message='im activations shape')
            weights1 = tf.Print(weights1, data = [weights1.get_shape()], message='weights1 shape')
            embed_im = tf.matmul(self.im_activations, weights1)
            embed_im = tf.Print(embed_im, data = [embed_im])
            word_dot = tf.mul(embed_im, word_embed)
            self.image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)    



class RecieverAgent:
    def __init__(self, vocab, activation_shape=[1000, 1], embedding_dim = 2):

        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.word = tf.placeholder(tf.int32, shape=())
        self.vgg = vgg16.Vgg16()
        self.im_activations = tf.placeholder("float", [2, 1000])
        self.image_scores = None
        self._build_image_selection_with_word()
        #self.embed_im = tf.placeholder(tf.float32, [2,2])

    def show_images(self, sess, comm_word, image1, image2):
        batch = np.concatenate((image1, image2), axis=0)
        image_scores = sess.run(self.image_scores, feed_dict={self.word : comm_word, self.im_activations : batch})
        selected = np.argmax(image_scores)
        return selected


    def _build_image_selection_with_word(self):
        # Build a tf graph that -
        # 1. image activations to embedding space
        # 2. Symbol to embedding space
        # 3. Dot product of symbol and each image

        with tf.name_scope('reciever'):
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim]))
            word_embed = vocab_embedding[self.word, :]
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            self.im_activations = tf.Print(self.im_activations, data = [self.im_activations.get_shape()], message='im activations shape')
            weights1 = tf.Print(weights1, data = [weights1.get_shape()], message='weights1 shape')
            embed_im = tf.matmul(self.im_activations, weights1)
            embed_im = tf.Print(embed_im, data = [embed_im])
            word_dot = tf.mul(embed_im, word_embed)
            self.image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)
