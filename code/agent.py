import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
import utils
import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class Agents:

    def __init__(self, vocab, image_embedding_dim = 2, embedding_dim = 2):

        self.vocab = vocab

        self.target_acts = tf.placeholder("float", [None, 1000])
        self.distractor_acts = tf.placeholder("float", [None, 1000])
        self.image_acts = tf.placeholder("float", [None, 2000])

        self.target = tf.placeholder(tf.int64, [None, 1])
        #self.target = tf.placeholder("float", [None, 1])
        self.image_embedding_dim = image_embedding_dim

        self.game_scores = tf.placeholder("float", [None, 1])

        self.vocab_scores = tf.placeholder("float", [None, len(self.vocab)])
        self.embedding_dim = embedding_dim
        #self.word = tf.placeholder(tf.int32, shape=())
        self.im_activations = tf.placeholder("float", [2, 1000])
        self.image_scores = tf.placeholder("float", [None, 2])

        print('now building the learning graph')
        self._build_learning_graph()
        print('finished building the learning graph')


    def _build_learning_graph(self):

        with tf.name_scope('sender'):

            ## Sender graph
            weights1 = tf.Variable(tf.random_normal([2000, self.image_embedding_dim], stddev=0.1))
            weights2 = tf.Variable(tf.random_normal([self.embedding_dim, len(self.vocab)], stddev=0.1))

            ordered_acts = tf.concat_v2([self.target_acts, self.distractor_acts], axis=1)
            h1 = tf.sigmoid(tf.matmul(ordered_acts, weights1))

            self.vocab_scores = tf.matmul(h1, weights2)

            word = tf.argmax(self.vocab_scores, axis=0)

            ## Reciever graph
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim]))
            print('here?')
            word_embed = tf.gather(vocab_embedding, word)
            print('here?')
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))

            print('coming till here')

            #image_acts = tf.concatenate([self.im1, self.im2], axis=0)
            #self.im_activations = tf.Print(self.image_acts, data = [self.im_activations.get_shape()], message='im activations shape')
            #weights1 = tf.Print(weights1, data = [weights1.get_shape()], message='weights1 shape')

            self.image_acts = tf.reshape(self.image_acts, [-1, 1000])
            embed_im = tf.matmul(self.image_acts, weights1)
            word_dot = tf.mul(embed_im, word_embed)
            self.image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)

            print(self.image_scores)
            self.reciever_selects = tf.argmax(self.image_scores, axis=0)
            #self.reciever_selects = tf.cast(self.reciever_selects, "float")

            self.game_scores = tf.reduce_mean(tf.square(self.reciever_selects - self.target))

            #self.loss = tf.reduce_sum(self.game_scores)

            self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            self.train_op = self.optimizer.minimize(self.game_scores)


    def show_images(self, sess, image_acts, target):
        target_acts = image_acts[target]
        distractor_acts = image_acts[1 - target]
        v_scores, _ = sess.run([self.vocab_scores, self.train_op], feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.image_acts : image_acts, self.target : target})
        comm_word = np.argmax(v_scores)
        return (target, comm_word)

    def train_sender(sess, samples, rewards):
        return 0
