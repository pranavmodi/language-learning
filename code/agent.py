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

            #self.vocab_scores = tf.Print(self.vocab_scores, [self.vocab_scores], message='vocab scores')

            self.word_probs = tf.squeeze(tf.nn.softmax(tf.div(self.vocab_scores, self.temperature)))
            self.twp = tf.transpose(self.word_probs)
            self.optimizer = tf.train.AdamOptimizer(0.5)

            self.twp = tf.Print(self.twp, [self.twp], message='word probs transpose')
            selected_word_prob = tf.gather(self.twp, self.word)
            self.sender_loss = -1 * tf.log(selected_word_prob) * self.reward
            #self.sender_loss = tf.Print(self.sender_loss, [self.sender_loss], message='sender loss')
            self.sender_train_op = self.optimizer.minimize(self.sender_loss)


            ## Reciever graph
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim], stddev=0.1))
            receiver_weights = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))

            self.word_embed = tf.gather(vocab_embedding, self.word)

            embed_im = tf.matmul(self.image_acts, receiver_weights)
            word_dot = tf.mul(embed_im, self.word_embed)
            image_scores = tf.reduce_sum(word_dot, 1, keep_dims=True)
            image_scores = tf.reshape(image_scores, [1, 2])a
            image_scores = tf.Print(image_scores, [image_scores], message='Image scores receiver')
            
            self.image_probs = tf.squeeze(tf.nn.softmax(tf.div(image_scores, self.temperature)))

            self.tip = tf.transpose(self.image_probs)
            self.tip = tf.Print(self.tip, [self.tip], message='image probability transpose')
            selected_image_prob = tf.gather(self.tip, self.selected_image)

            self.receiver_loss = -1 * tf.log(selected_image_prob) * self.reward

            self.receiver_train_op = self.optimizer.minimize(self.receiver_loss)

            #cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.reciever_selects))

            #self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
            #optimizer = tf.train.AdamOptimizer(0.5)

            #correct_prediction = tf.equal(self.reciever_selects, self.target)
            #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            #print(accuracy)

            #self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
            #self.train_op = self.optimizer.minimize(accuracy)


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

        print(target_class, reward)

        ## Run learning operations
        sender_train_op, receiver_train_op, sender_loss, receiver_loss = sess.run([self.sender_train_op, self.receiver_train_op, self.sender_loss, self.receiver_loss], feed_dict={self.image_acts : image_acts, self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.reward : reward, self.word : word, self.selected_image : selected_image})

        return reward

    def train_sender(sess, samples, rewards):
        return 0
