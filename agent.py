import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
#import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class Agents:

    def __init__(self, vocab, image_embedding_dim, word_embedding_dim,
                 learning_rate, temperature=10, batch_size=32):

        self.vocab = vocab
        self.target_acts = tf.placeholder("float",
                                          [None, 1000], name="target_acts")
        self.distractor_acts = tf.placeholder("float",
                                              [None, 1000], name="distractor_acts")
        self.image_acts = tf.placeholder("float",
                                         [None, 2000], name="combined_acts")

        self.image_embedding_dim = image_embedding_dim
        self.batch_size = batch_size

        self.word_embedding_dim = word_embedding_dim

        self.word = tf.placeholder(tf.int32, [1, None], "word")
        self.selection = tf.placeholder(tf.int32, [1, None], "selection")
        self.reward = tf.placeholder(tf.float32, [None, 1], "reward")

        self.epsilon = 0.1
        self.temperature = temperature

        self.learning_rate = learning_rate

        print('now building the learning graph')
        self._build_learning_graph()
        print('finished building the learning graph')


    def _build_learning_graph(self):

        with tf.name_scope('sender'):

            ## Sender graph
            with tf.name_scope('ordered_embed'):
                t_weights = tf.Variable(tf.random_normal([1000, self.image_embedding_dim], stddev=0.01), name = "sender_t")
                t_bias = tf.Variable(tf.zeros([1, self.image_embedding_dim]), trainable=True, name="t_bias")
                d_weights = tf.Variable(tf.random_normal([1000, self.image_embedding_dim], stddev=0.01), name = "sender_d")
                d_bias = tf.Variable(tf.zeros([1, self.image_embedding_dim]), trainable=True, name="d_bias")

                self.t_embed = tf.sigmoid(tf.matmul(self.target_acts, t_weights) + t_bias, name = "t_embed") ## Add bias?
                self.d_embed = tf.sigmoid(tf.matmul(self.distractor_acts, d_weights) + d_bias, name = "d_embed") ## Add bias?
                self.ordered_embed = tf.concat([self.t_embed, self.d_embed], axis=1)

            with tf.name_scope('word_probs'):
                gsi_embed = tf.Variable(tf.random_normal([(2 * self.image_embedding_dim), len(self.vocab)], stddev=0.01), name = "gsi_embed")
                self.vocab_scores = tf.matmul(self.ordered_embed, gsi_embed, name="vocab_scores")
                self.word_probs = tf.nn.softmax(tf.div(self.vocab_scores, self.temperature), name="word_probs")
                #self.word_probs = tf.Print(self.word_probs, [tf.shape(self.word_probs)], message='word probs shape')

            with tf.name_scope('sender_optimization'):
                self.sender_optimizer = tf.train.AdamOptimizer(self.learning_rate)
                word_probs_flattened = tf.reshape(self.word_probs, [-1])
                selected_inds = tf.range(0, tf.shape(self.word_probs)[0]) * len(self.vocab) + self.word
                selected_word_prob = tf.gather(tf.reshape(self.word_probs, [-1]), selected_inds)
                #selected_word_prob = tf.Print(selected_word_prob, [tf.shape(selected_word_prob)], message='selected word prob shape')

                #self.reward = tf.Print(self.reward, [tf.shape(self.reward)], message='reward shape')
                self.sender_loss = tf.reduce_mean(-1 * tf.multiply(tf.transpose(tf.log(selected_word_prob)), self.reward, name="sender_loss"))
                #grads_and_vars = self.sender_optimizer.compute_gradients(self.sender_loss)

                #self.sender_loss = tf.Print(self.sender_loss, [self.sender_loss], message='sender loss')
                #self.sender_loss = tf.Print(self.sender_loss, [tf.shape(self.sender_loss)], message='shape of sender loss')
                #gvp = [tf.Print(gv[0], [tf.shape(gv[0])], 'GV shape: ') for gv in grads_and_vars]
                self.sender_train_op = self.sender_optimizer.minimize(self.sender_loss)

        with tf.name_scope('receiver'):
            ## Reciever graph

            with tf.name_scope('image_embed'):
                receiver_weights = tf.Variable(tf.random_normal([1000, self.word_embedding_dim], stddev=0.01))
                receiver_bias = tf.Variable(tf.zeros([1, self.word_embedding_dim]), trainable=True, name="receiver_bias")
                ti_acts = tf.transpose(self.image_acts)
                im1 = tf.transpose(tf.gather(ti_acts, tf.range(0,1000)))
                im2 = tf.transpose(tf.gather(ti_acts, tf.range(1000,2000)))

                im1_embed = tf.matmul(im1, receiver_weights) + receiver_bias
                print('im1 embed', im1_embed)

                im2_embed = tf.matmul(im2, receiver_weights) + receiver_bias

            with tf.name_scope('image_select'):
                vocab_embedding = tf.Variable(tf.random_normal([len(self.vocab), self.word_embedding_dim], stddev=0.01))
                self.word_embed = tf.squeeze(tf.gather(vocab_embedding, self.word))
                print('word embed', self.word_embed)
                #self.word_embed = tf.Print(self.word_embed, [tf.shape(self.word_embed)], message='word embed shape')
                #im1_embed = tf.Print(im1_embed, [tf.shape(im1_embed)], message='image 1 embed shape')
                # im1_dot = tf.tensordot(im1_embed, self.word_embed, axes=1)
                # im1_dot = tf.Print(im1_dot, [tf.shape(im1_dot)], message='After dotting the shape')
                # im2_dot = tf.tensordot(im2_embed, self.word_embed, axes=1)
                # im1_scores = tf.reduce_sum(im1_dot, 0)
                # im1_scores = tf.reshape(im1_scores, [-1, 1])
                # im2_scores = tf.reduce_sum(im2_dot, 0)
                # im2_scores = tf.reshape(im2_scores, [-1, 1])

                im1_dot = tf.multiply(im1_embed, self.word_embed)
                #im1_dot = tf.Print(im1_dot, [tf.shape(im1_dot)], message='im1 dot shape')
                im1_sum = tf.reduce_sum(im1_dot, 1)
                #im1_sum = tf.Print(im1_sum, [tf.shape(im1_sum)], message='im1_sum shape')
                im1_scores = tf.reshape(im1_sum, [-1, 1])
                #im1_scores = tf.Print(im1_scores, [tf.shape(im1_scores)], message='im1_scores shape')
                
                im2_dot = tf.multiply(im2_embed, self.word_embed)
                im2_sum = tf.reduce_sum(im2_dot, 1)
                im2_scores = tf.reshape(im2_sum, [-1, 1])
                image_scores = tf.concat([im1_scores, im2_scores], axis=1)
                #image_scores = tf.Print(image_scores, [tf.shape(image_scores)], message='image scores shape')

                self.image_probs = tf.squeeze(tf.nn.softmax(tf.div(image_scores, 10000)))
                #self.image_probs = tf.Print(self.image_probs, [tf.shape(self.image_probs)], message='Image probs shapeeeeeeeee')
                image_probs_flattened = tf.reshape(self.image_probs, [-1])
                selected_inds = tf.range(0, tf.shape(self.image_probs)[0]) * len(self.vocab) + self.selection
                selected_image_prob = tf.gather(tf.reshape(self.image_probs, [-1]), selected_inds)
                #selected_image_prob = tf.Print(selected_image_prob, [tf.shape(selected_image_prob)], message='selected image probs shape')

            with tf.name_scope('receiver_optimize'):
                self.receiver_optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.receiver_loss = tf.reduce_mean(-1 * tf.log(selected_image_prob) * self.reward)
                #self.receiver_loss = tf.Print(self.receiver_loss, [self.receiver_loss], message='receiver loss')
                self.receiver_train_op = self.receiver_optimizer.minimize(self.receiver_loss)

        with tf.name_scope('sender_weights'):
            self.plot_var_summary('t_weights', t_weights)
            self.plot_var_summary('d_weights', d_weights)
            self.plot_var_summary('gsi_embed', gsi_embed)
            self.plot_var_summary('t_bias', t_bias)
            self.plot_var_summary('d_bias', d_bias)
            self.plot_var_summary('gsi_embed', gsi_embed)

        with tf.name_scope('receiver_weights'):
            self.plot_var_summary('receiver_weights', receiver_weights)
            self.plot_var_summary('receiver_bias', receiver_bias)
            self.plot_var_summary('vocab_embedding', vocab_embedding)

        with tf.name_scope('sender_activations'):
            self.plot_var_summary('t_embed', self.t_embed)
            self.plot_var_summary('d_embed', self.d_embed)
            self.plot_var_summary('ordered_embed', self.ordered_embed)
            self.plot_var_summary('vocab_scores', self.vocab_scores)
            tf.summary.scalar('sender_loss', self.sender_loss)

        with tf.name_scope('receiver_activations'):
            self.plot_var_summary('im1_embed', im1_embed)
            self.plot_var_summary('im2_embed', im2_embed)
            self.plot_var_summary('im1_dot', im1_dot)
            self.plot_var_summary('im2_dot', im2_dot)
            tf.summary.scalar('receiver_loss', self.receiver_loss)
            
        with tf.name_scope('sender_gradients'):
            sgrads = tf.gradients(self.sender_loss, tf.trainable_variables())
            sgrads = list(zip(sgrads, tf.trainable_variables()))
            for grad, var in sgrads:
                #print('one of the vars', var.name)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradient', grad)


        with tf.name_scope('receiver_gradients'):
            rgrads = tf.gradients(self.receiver_loss, tf.trainable_variables())
            rgrads = list(zip(rgrads, tf.trainable_variables()))
            for grad, var in rgrads:
                #print('one of the vars', var.name)
                if grad is not None:
                    tf.summary.histogram(var.name + '/gradient', grad)

        self.summary = tf.summary.merge_all()


    def plot_var_summary(self, varname, var):
        var_mean = tf.reduce_mean(var)
        var_std = tf.sqrt(tf.reduce_mean(tf.square(var - var_mean)))
        tf.summary.scalar(varname + '_mean', var_mean)
        tf.summary.scalar(varname + '_std', var_std)
        tf.summary.histogram(varname + '_hist', var)


    def show_images(self, sess, image_acts, target_acts, distractor_acts, target, target_class):

        word_probs = sess.run(self.word_probs, feed_dict={self.target_acts : target_acts, self.distractor_acts : distractor_acts})[0]
        print('word probs', word_probs)
        word_probs = word_probs + [0.1, 0.1]
        word_probs = word_probs / np.sum(word_probs)
        print('new word probs', word_probs)
        
        word = np.random.choice(np.arange(len(self.vocab)), p=word_probs)
        word = np.reshape(np.array(word), [1, -1])

        ## Receiver select images op
        image_probs = sess.run(self.image_probs, feed_dict={self.image_acts : image_acts, self.word : word})
        #print('image probs', image_probs)
        selection = np.random.choice(np.arange(2), p=image_probs)

        reward = -1.0
        if selection == target:
            reward = 1.0

        return reward, word, selection, word_probs, image_probs


    def update(self, sess, batch):
        acts_batch, target_acts, distractor_acts, word_probs_batch, \
            image_probs_batch, target_batch, word_batch, selection_batch, reward_batch = map(lambda x: np.squeeze(np.array(x)), zip(*batch))

        reward_batch = np.reshape(reward_batch, [-1, 1])
        selection_batch = np.reshape(selection_batch, [1, -1])
        word_batch = np.reshape(word_batch, [1, -1])
        target_acts = np.reshape(target_acts, [-1, 1000])
        distractor_acts = np.reshape(distractor_acts, [-1, 1000])
        acts_batch = np.reshape(acts_batch, [-1, 2000])

        ret_tup = sess.run([self.sender_train_op, self.receiver_train_op,self.sender_loss,
                            self.receiver_loss, self.summary],
                           feed_dict={self.image_acts : acts_batch, self.target_acts : target_acts,
                                      self.distractor_acts : distractor_acts, self.reward : reward_batch,
                                      self.word : word_batch, self.selection : selection_batch})

        sender_train_op, receiver_train_op, sender_loss, receiver_loss, summary = ret_tup

        return summary


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
