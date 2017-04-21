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

        #self.target = tf.placeholder(tf.int64, [None, 1])
        self.target = tf.placeholder("float", [None, 1])
        self.image_embedding_dim = image_embedding_dim

        self.game_scores = tf.placeholder("float", [None, 1])

        self.word_probs = tf.placeholder("float", [None, len(self.vocab)])
        self.embedding_dim = embedding_dim

        self.word = tf.placeholder(tf.int32, [], "word")
        self.reward = tf.placeholder(tf.float32, [], "reward")

        self.im_activations = tf.placeholder("float", [2, 1000])
        self.image_scores = tf.placeholder("float", [None, 2])
        self.epsilon = 0.1

        print('now building the learning graph')
        #self._build_learning_graph()
        self._build_learning_graph()
        print('finished building the learning graph')

    def _build_trial_graph(self):

        with tf.name_scope('trial'):

            ## Sender graph
            weights1 = tf.Variable(tf.random_normal([1000, self.image_embedding_dim], stddev=0.1), name = "sender_w1", trainable="true")
            weights2 = tf.Variable(tf.random_normal([self.embedding_dim, len(self.vocab)], stddev=0.1), name = "sender_w2", trainable="true")

            h1 = tf.sigmoid(tf.matmul(self.target_acts, weights1), name = "h1")
            vocab_scores = tf.matmul(h1, weights2, name = "vocab_scores")

            self.softmax_probs = tf.nn.softmax(self.word_probs)

            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax_probs, labels=self.target), name = "cross_entropy")
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)



    def call_trial(self, sess, image_acts, target):
        target_acts = image_acts[target]
        distractor_acts = image_acts[1 - target]

        target = np.reshape(target, [-1, 1])
        image_acts = np.reshape(image_acts, [-1, 2000])
        word_probs = sess.run([self.word_probs], feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.image_acts : image_acts, self.target : target})

        if random.random() < self.epsilon:
            return random_action
        word = np.random.choice(word_probs.shape[0], 1, p=word_probs)[0]

        print(v_scores, word)


    def _build_learning_graph(self):

        with tf.name_scope('sender'):

            ## Sender graph
            weights1 = tf.Variable(tf.random_normal([2000, self.image_embedding_dim], stddev=0.1), name = "sender_w1")
            weights2 = tf.Variable(tf.random_normal([self.embedding_dim, len(self.vocab)], stddev=0.1))

            ordered_acts = tf.concat_v2([self.target_acts, self.distractor_acts], axis=1)
            h1 = tf.sigmoid(tf.matmul(ordered_acts, weights1))

            self.word_probs = tf.nn.softmax(tf.matmul(h1, weights2))
            self.optimizer = tf.train.AdamOptimizer(0.5)
            selected_word_prob = tf.gather(self.word_probs, self.word)
            self.sender_loss = -tf.log(selected_word_prob) * self.reward
            self.sender_train_op = self.optimizer.minimize(self.sender_loss)

            ## Reciever graph
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim]))
            weights1 = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))

            self.word_embed = tf.gather(vocab_embedding, self.word)
            self.image_acts = tf.reshape(self.image_acts, [-1, 1000])

            embed_im = tf.matmul(self.image_acts, weights1)
            word_dot = tf.mul(embed_im, self.word_embed)
            self.image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)

            self.image_probs = tf.nn.softmax(self.image_scores)

            #self.reciever_selects = tf.argmax(self.image_scores, axis=0)

            #self.reciever_selects = tf.cast(self.reciever_selects, tf.float32)

            #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.reciever_selects))

            #self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            #optimizer = tf.train.AdamOptimizer(0.5)

            #correct_prediction = tf.equal(self.reciever_selects, self.target)
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #print(accuracy)

            #self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            #self.train_op = self.optimizer.minimize(accuracy)


    def show_images(self, sess, image_acts, target):
        target_acts = image_acts[target]
        distractor_acts = image_acts[1 - target]

        word_probs = sess.run(self.word_probs, feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts})[0]

        print(word_probs)

        self.word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)

        if target==self.word:
            self.reward = 1.0
        else:
            self.reward = 0.0

        ## Now train the sender to select cat pics
        


        #print(self.word)
        #v_scores, _ = sess.run([self.word_probs, self.train_step], feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.image_acts : image_acts, self.target : target})
        #comm_word = np.argmax(v_scores)
        return (target, self.word)

    def train_sender(sess, samples, rewards):
        return 0
