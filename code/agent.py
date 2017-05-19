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

        self.target_acts = tf.placeholder("float", [None, 1000], name="target_acts")
        self.distractor_acts = tf.placeholder("float", [None, 1000], name="distractor_acts")
        self.image_acts = tf.placeholder("float", [None, 2000], name="combined_acts")

        #self.target = tf.placeholder(tf.int32, [None, 1])
        self.image_embedding_dim = image_embedding_dim

        self.word_probs = tf.placeholder("float", [None, len(self.vocab)], name="word_probs")
        self.embedding_dim = embedding_dim

        self.word = tf.placeholder(tf.int32, [None, 1], "word")
        self.selection = tf.placeholder(tf.int32, [None, 1], "selection")
        self.reward = tf.placeholder(tf.float32, [None, 1], "reward")

        self.image_probs = tf.placeholder("float", [None, 2], name="image_probs")
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

            #distractor = 1 - self.target
            #target_indices = tf.map_fn(lambda x: tf.range(x * 1000, (x + 1) * 1000), self.target)
            #distractor_indices = tf.map_fn(lambda x: tf.range(x * 1000, (x + 1) * 1000), self.distractor)

            #target_acts = tf.map_fn(lambda x: self.image_acts, axis=0)
            #distractor_acts = tf.map_fn(lambda x: self.image_acts, axis=0)

            t_embed = tf.sigmoid(tf.matmul(self.target_acts, t_weights), name = "t_embed")
            d_embed = tf.sigmoid(tf.matmul(self.distractor_acts, d_weights), name = "d_embed")
            ordered_embed = tf.concat_v2([t_embed, d_embed], axis=1)
            gsi_embed = tf.Variable(tf.random_normal([(2 * self.image_embedding_dim), len(self.vocab)], stddev=0.1), name = "gsi_embed")

            self.vocab_scores = tf.matmul(ordered_embed, gsi_embed, name="vocab_scores")
            self.vocab_scores = tf.Print(self.vocab_scores, [self.vocab_scores], message='sender vocab scores')
            self.word_probs = tf.squeeze(tf.nn.softmax(tf.div(self.vocab_scores, self.temperature)), name="word_probs")
            self.twp = tf.transpose(self.word_probs, name="twp")
            self.sender_optimizer = tf.train.AdamOptimizer(0.2)
            self.twp = tf.Print(self.twp, [tf.shape(tf.transpose(self.twp))], message='word probs transpose')
            selected_word_prob = tf.gather(self.twp, self.word)
            self.sender_loss = tf.reduce_sum(-1 * tf.log(selected_word_prob) * self.reward, name="sender_loss")
            self.sender_loss = tf.Print(self.sender_loss, [self.sender_loss], message='sender loss')
            self.sender_train_op = self.sender_optimizer.minimize(self.sender_loss)
            
        with tf.name_scope('reciever'):
            ## Reciever graph
            vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.embedding_dim], stddev=0.1))
            receiver_weights = tf.Variable(tf.random_normal([1000, self.embedding_dim], stddev=0.1))
            receiver_bias = tf.Variable(tf.zeros([1, self.embedding_dim]), trainable=True, name="receiver_bias")

            self.word_embed = tf.squeeze(tf.gather(vocab_embedding, self.word))
            #self.word_embed = tf.Print(self.word_embed, [tf.shape(self.word_embed)], message='word embedded shape')

            ti_acts = tf.transpose(self.image_acts)
            im1 = tf.transpose(tf.gather(ti_acts, tf.range(0,1000)))
            im2 = tf.transpose(tf.gather(ti_acts, tf.range(1000,2000)))

            #im2 = tf.Print(im2, [tf.shape(im2)], message='im2 image shape')
            #embed_im = tf.matmul(self.image_acts, receiver_weights) + receiver_bias

            im1_embed = tf.matmul(im1, receiver_weights) + receiver_bias
            im2_embed = tf.matmul(im2, receiver_weights) + receiver_bias

            embed_im = tf.concat_v2([im1_embed, im2_embed], axis=0)
            im1_dot = tf.mul(im1_embed, self.word_embed)
            im2_dot = tf.mul(im2_embed, self.word_embed)

            im1_scores = tf.reduce_sum(im1_dot, 1)
            im1_scores = tf.reshape(im1_scores, [-1, 1])
            im2_scores = tf.reduce_sum(im2_dot, 1)
            im2_scores = tf.reshape(im2_scores, [-1, 1])

            #im1_scores = tf.Print(im1_scores, [tf.shape(im1_scores)], message='im1 values')
            #image_scores = tf.reshape(image_scores, [-1, 2])
            image_scores = tf.concat_v2([im1_scores, im2_scores], axis=1)
            #image_scores = tf.Print(image_scores, [tf.shape(image_scores)], message='Image scores shape')

            self.image_probs = tf.squeeze(tf.nn.softmax(tf.div(image_scores, 10000)))
            #self.image_probs = tf.Print(self.image_probs, [tf.shape(self.image_probs)], message='Image probs shape')
            self.tip = tf.transpose(self.image_probs)
            #self.tip = tf.Print(self.tip, [self.tip], message='image probability transpose')
            selected_image_prob = tf.gather(self.tip, self.selection)
            self.receiver_optimizer = tf.train.AdamOptimizer(0.2)
            self.receiver_loss = tf.reduce_sum(-1 * tf.log(selected_image_prob) * self.reward)
            self.receiver_loss = tf.Print(self.receiver_loss, [self.receiver_loss], message='receiver loss')
            self.receiver_train_op = self.receiver_optimizer.minimize(self.receiver_loss)

    def show_images(self, sess, image_acts, target_acts, distractor_acts, target, target_class):
        #target_acts = image_acts[target]
        #distractor_acts = image_acts[1 - target]

        word_probs = sess.run(self.word_probs, feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts})
        print('word probs', word_probs)
        word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)
        word = np.reshape(np.array(word), [-1, 1])
        #word_text = self.vocab[word, :]

        ## Receiver select images op
        image_probs = sess.run(self.image_probs, feed_dict={self.image_acts : image_acts, self.word : word})
        print('image probs', image_probs)
        selection = np.random.choice(np.arange(2), p=image_probs)

        reward = -1.0
        if selection == target:
            reward = 1.0

        ## Run learning operations
        #sender_train_op, receiver_train_op, sender_loss, receiver_loss = sess.run([self.sender_train_op, self.receiver_train_op, self.sender_loss, self.receiver_loss], feed_dict={self.image_acts : image_acts, self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.reward : reward, self.word : word, self.selection : selection})

        return reward, word, selection


    def update(self, sess, batch):
        acts_batch, target_acts, distractor_acts, target_batch, word_batch, selection_batch, reward_batch = map(lambda x: np.squeeze(np.array(x)), zip(*batch))

        reward_batch = np.reshape(reward_batch, [-1, 1])
        selection_batch = np.reshape(selection_batch, [-1, 1])
        word_batch = np.reshape(word_batch, [-1, 1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts_batch = np.reshape(acts_batch, [-1, 2000])

        ## Update sender
        sender_train_op, receiver_train_op, sender_loss, receiver_loss = sess.run([self.sender_train_op, self.receiver_train_op, self.sender_loss, self.receiver_loss], feed_dict={self.image_acts : acts_batch, self.target_acts : target_acts, self.distractor_acts : distractor_acts, self.reward : reward_batch, self.word : word_batch, self.selection : selection_batch})

        ## Update receiver




    def test_receiver(self, sess, image_acts, word, target_ind, target_class):
        image_probs = sess.run(self.image_probs, feed_dict={self.image_acts : image_acts, self.word : word})
        print('Image probs', image_probs)
        selected_image = np.random.choice(np.arange(2), p=image_probs)

        reward = -1.0
        if selected_image == target_ind:
            reward = 1.0

        print(target_class, target_ind, selected_image, reward)
        receiver_train_op, receiver_loss = sess.run([self.receiver_train_op, self.receiver_loss], feed_dict={self.image_acts : image_acts, self.reward : reward, self.word : word, self.selection : selected_image})

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
