import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
import utils
import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class Agents:

    def __init__(self, vocab, image_embedding_dim = 2, embedding_dim = 2, temperature=10):

        self.vocab = vocab

        self.target_acts = tf.placeholder("float", [None, 1000])
        self.distractor_acts = tf.placeholder("float", [None, 1000])
        self.image_acts = tf.placeholder("float", [2, 1000])

        #self.target = tf.placeholder(tf.int64, [None, 1])
        self.target = tf.placeholder("float", [None, 1])
        self.image_embedding_dim = image_embedding_dim

        self.game_scores = tf.placeholder("float", [None, 1])

        #self.word_probs = tf.placeholder("float", [len(self.vocab), 1])
        self.word_probs = tf.placeholder("float", [None, len(self.vocab)])
        self.embedding_dim = embedding_dim

        self.word = tf.placeholder(tf.int32, [], "word")
        self.selected_image = tf.placeholder(tf.int32, [], "image")
        self.reward = tf.placeholder(tf.float32, [], "reward")

        self.im_activations = tf.placeholder("float", [2, 1000])
        self.image_scores = tf.placeholder("float", [None, 2])
        self.epsilon = 0.1
        self.temperature = temperature

        print('now building the learning graph')
        #self._build_learning_graph()
        self._build_learning_graph()
        print('finished building the learning graph')


    def _build_learning_graph(self):

        with tf.name_scope('sender'):

            ## Sender graph
            t_weights = tf.Variable(tf.random_normal([1000, self.image_embedding_dim], stddev=0.1), name = "sender_t")
            d_weights = tf.Variable(tf.random_normal([1000, self.image_embedding_dim], stddev=0.1), name = "sender_d")

            t_embed = tf.sigmoid(tf.matmul(self.target_acts, t_weights))
            d_embed = tf.sigmoid(tf.matmul(self.distractor_acts, d_weights))
            ordered_embed = tf.concat_v2([t_embed, d_embed], axis=1)
            gsi_embed = tf.Variable(tf.random_normal([(2 * self.image_embedding_dim), len(self.vocab)], stddev=0.1))

            self.vocab_scores = tf.matmul(ordered_embed, gsi_embed)
            self.word_probs = tf.squeeze(tf.nn.softmax(tf.div(self.vocab_scores, self.temperature)))
            self.twp = tf.transpose(self.word_probs)
            self.sender_optimizer = tf.train.AdamOptimizer(0.1)
            self.twp = tf.Print(self.twp, [self.twp], message='word probs transpose')
            selected_word_prob = tf.gather(self.twp, self.word)
            self.sender_loss = -1 * tf.log(selected_word_prob) * self.reward
            self.sender_train_op = self.sender_optimizer.minimize(self.sender_loss)

        with tf.name_scope('sender'):
            ## Reciever graph
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim], stddev=0.1))
            receiver_weights = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            receiver_bias = tf.Variable(tf.zeros([1, self.embedding_dim]), trainable=True, name="receiver_bias")

            self.word_embed = tf.gather(vocab_embedding, self.word)
            self.word_embed = tf.Print(self.word_embed, [self.word_embed], message='word embedded scores')

            embed_im = tf.matmul(self.image_acts, receiver_weights) + receiver_bias
            embed_im = tf.Print(embed_im, [embed_im], message='embedded images values')
            word_dot = tf.mul(embed_im, self.word_embed)
            image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)
            #image_scores = tf.Print(image_scores, [image_scores], message='embedded images values')
            image_scores = tf.reshape(image_scores, [1, 2])
            image_scores = tf.Print(image_scores, [image_scores], message='Image scores receiver')

            self.image_probs = tf.squeeze(tf.nn.softmax(tf.div(image_scores, 10000)))

            self.tip = tf.transpose(self.image_probs)
            self.tip = tf.Print(self.tip, [self.tip], message='image probability transpose')
            selected_image_prob = tf.gather(self.tip, self.selected_image)
            self.receiver_optimizer = tf.train.AdamOptimizer(0.2)
            self.receiver_loss = -1 * tf.log(selected_image_prob) * self.reward
            self.receiver_train_op = self.receiver_optimizer.minimize(self.receiver_loss)

    def show_images(self, sess, image_acts, target, target_class):
        target_acts = image_acts[target]
        distractor_acts = image_acts[1 - target]

        word_probs = sess.run(self.word_probs, feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts})
        print('word probs', word_probs)
        word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)
        word_text = self.vocab[word]

        ## Receiver select images op
        image_probs = sess.run(self.image_probs, feed_dict={self.image_acts : image_acts, self.word : word})
        print('Image probs', image_probs)
        selected_image = np.random.choice(np.arange(2), p=image_probs)

        reward = -1.0
        if selected_image == target:
            reward = 1.0

        ## Run learning operations
        sender_train_op, receiver_train_op, sender_loss, receiver_loss = sess.run([self.sender_train_op, self.receiver_train_op, self.sender_loss, self.receiver_loss], feed_dict={self.image_acts : image_acts, self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.reward : reward, self.word : word, self.selected_image : selected_image})

        return reward, word_text

    def test_receiver(self, sess, image_acts, word, target_ind, target_class):
        image_probs = sess.run(self.image_probs, feed_dict={self.image_acts : image_acts, self.word : word})
        print('Image probs', image_probs)
        selected_image = np.random.choice(np.arange(2), p=image_probs)

        reward = -1.0
        if selected_image == target_ind:
            reward = 1.0

        print(target_class, target_ind, selected_image, reward)
        receiver_train_op, receiver_loss = sess.run([self.receiver_train_op, self.receiver_loss], feed_dict={self.image_acts : image_acts, self.reward : reward, self.word : word, self.selected_image : selected_image})

        return reward


    def test_sender(self, sess, image_acts, target, target_class):
        target_acts = image_acts[target]
        distractor_acts = image_acts[1 - target]

        word_probs = sess.run(self.word_probs, feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts})
        print('\nword probs', word_probs)
        word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)
        word_text = self.vocab[word]

        reward = -1.0
        if target_class == 'dog':
            if word_text == 'Dogword':
                reward = 1.0
        elif target_class == 'cat':
            if word_text == 'Catword':
                reward = 1.0

        sender_train_op, sender_loss = sess.run([self.sender_train_op, self.sender_loss], feed_dict={self.image_acts : image_acts, self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.reward : reward, self.word : word})

        return reward, word_text
