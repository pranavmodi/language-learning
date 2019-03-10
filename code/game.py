import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import agent
import env
import skimage
import random
import tensorflow as tf
import sys
import argparse
import yaml
sys.path.append('tensorflow-vgg/')
from vgg16 import Vgg16
from collections import deque, namedtuple


def load_model(npy_path):
    v = vgg16(npy_path)


def some_shit():

    save_path = os.path.join('..', 'model', run_num + '.ckpt')
    load_model = False
    load_path = os.path.join('..', 'model', '16.ckpt')

    num_words = 2
    vocab = ['W'+str(i) for i in range(num_words)]

    #vocab = ['Catword', 'Dogword']
    embed_dim = 2
    print(vocab)


def get_image_activations(sess, vgg, image, placeholder):
    #image_pl = tf.placeholder("float32", [1, 224, 224, 3])
    batch = image.reshape((1, 224, 224, 3))
    feed_dict = {placeholder: batch}
    
    with tf.name_scope("content_vgg"):
        fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
    
    return(fc8)


def shuffle_image_activations(im_acts):
    reordering = np.array(range(len(im_acts)))
    random.shuffle(reordering)
    target_ind = np.argmin(reordering)
    shuffled = im_acts[reordering]
    return (shuffled, target_ind)


# def make_epsilon_greedy_policy(estimator, nA):
#     """
#     Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

#     Args:
#         estimator: An estimator that returns q values for a given state
#         nA: Number of actions in the environment.

#     Returns:
#         A function that takes the (sess, observation, epsilon) as an argument and returns
#         the probabilities for each action in the form of a numpy array of length nA.

#     """
#     def policy_fn(sess, observation, epsilon):
#         A = np.ones(nA, dtype=float) * epsilon / nA
#         q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
#         best_action = np.argmax(q_values)
#         A[best_action] += (1.0 - epsilon)
#         return A
#     return policy_fn


def run_game(config):

    tf.reset_default_graph()
    image_embedding_dim = config['image_embedding_dim']
    word_embedding_dim = config['word_embedding_dim']
    data_dir = config['data_dir']
    image_dirs = config['image_dirs']
    vocab = config['vocab']
    log_path = config['log_path']
    vgg16_weights = config['vgg16_weights']
    learning_rate = config['learning_rate']
    
    agents = agent.Agents(vocab, image_embedding_dim,
                          word_embedding_dim, learning_rate, temperature=10, batch_size=2)
    game = env.Environment(data_dir, image_dirs, 2)

    writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
    saver = tf.train.Saver()
    save_every = 10

    ## Run the iterations of the game
    iterations = 20000
    mini_batch_size = 2

    num_classes = len(image_dirs)

    wins = 0
    losses = 0

    update_estimators_every = 50

    with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
        vgg = Vgg16(vgg16_weights)

        image_pl = tf.placeholder("float32", [1, 224, 224, 3])
        vgg.build(image_pl)
        if load_model==True:
            saver.restore(sess, load_path)
            print("Restored model from : ", load_path)
        else:
            sess.run(tf.global_variables_initializer())
            print('Initialized the model')

        batch = []
        Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts", "word_probs", "image_probs", "target", "word", "selection", "reward"])
        tot_reward = 0
        for i in range(iterations):

            print("\rEpisode {}/{}".format(i, iterations), end="")
            sys.stdout.flush()

            if i % 10 == 0:
                print('last 10 interations performance ', tot_reward)
                tot_reward = 0

            target_image, distractor_image = game.get_images()
            target_class = game.target_class
            target_acts = get_image_activations(sess, vgg, target_image, image_pl)
            distractor_acts = get_image_activations(sess, vgg, distractor_image, image_pl)

            reordering = np.array([0,1])
            random.shuffle(reordering)
            target = np.where(reordering==0)[0]

            img_array = [target_acts, distractor_acts] 
            i1, i2 = [img_array[reordering[i]] for i, img in enumerate(img_array)]

            shuffled_acts = np.concatenate([i1, i2], axis=1)

            ## for Sender - take action in reinforcement learning terms

            reward, word, selection, word_probs, image_probs = agents.show_images(sess, shuffled_acts, target_acts, distractor_acts, target, target_class)
            print(word_probs)

            batch.append(Game(shuffled_acts, target_acts, distractor_acts, word_probs, image_probs, target, word, selection, reward))

            if len(batch) > mini_batch_size:
                batch.pop(0)

            if (i+1) % mini_batch_size == 0:
                print('updating the agent weights')
                summary = agents.update(sess, batch)
                writer.add_summary(summary, i)
            # if (i+1) % save_every == 0:
            #     save_path = saver.save(sess, save_path)
            #     print("Model saved in file: %s" % save_path)

            #reward, word_text = agents.test_sender(sess, shuffled_acts, target, target_class)
            print(target_class, reward)
            #reward = agents.test_receiver(sess, shuffled_acts, word, target_ind, target_class)
            tot_reward += reward
            selection = 0
            #agents.call_trial(sess, img_array, target_ind)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True)
    args = parser.parse_args()
    conf = args.conf

    with open(conf) as g:
        config = yaml.load(g)

    print(config)
    run_game(config)

if __name__ == '__main__':
    main()
