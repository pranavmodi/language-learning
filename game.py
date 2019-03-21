import sys
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from agent_tf2 import Agents
import env
import random
import tensorflow as tf
import sys
import argparse
import yaml
from tensorflow.keras.applications.vgg16 import VGG16
from collections import deque, namedtuple


def shuffle_image_activations(im_acts):
    reordering = np.array(range(len(im_acts)))
    random.shuffle(reordering)
    target_ind = np.argmin(reordering)
    shuffled = im_acts[reordering]
    return (shuffled, target_ind)


def run_game(config):

    image_embedding_dim = config['image_embedding_dim']
    word_embedding_dim = config['word_embedding_dim']
    data_dir = config['data_dir']
    image_dirs = config['image_dirs']
    vocab = config['vocab']
    log_path = config['log_path']
    model_weights = config['model_weights']
    learning_rate = config['learning_rate']
    load_model = config['load_model'] == 'True'

    iterations = config['iterations']
    mini_batch_size = config['mini_batch_size']
    
    agents = Agents(vocab,
                    image_embedding_dim,
                    word_embedding_dim,
                    learning_rate,
                    temperature=10,
                    batch_size=2)

    environ = env.Environment(data_dir, image_dirs, 2)

    # writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
    # saver = tf.train.Saver()
    # save_every = 10

    wins = 0
    losses = 0
    model = VGG16()


    batch = []
    Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts", "word_probs",
                               "image_probs", "target", "word", "selection", "reward"])
    tot_reward = 0
    for i in range(iterations):

        print('Episode {}/{}'.format(i, iterations), end='\n')

        if i % 10 == 0:
            print('last 10 interations performance ', tot_reward)
            tot_reward = 0

        target_image, distractor_image = environ.get_images()
        target_image = target_image.reshape((1, 224, 224, 3))
        distractor_image = distractor_image.reshape((1, 224, 224, 3))

        target_class = environ.target_class

        td_images = np.vstack([target_image, distractor_image])
        td_acts = model.predict(td_images)

        target_acts = td_acts[0].reshape((1, 1000))
        distractor_acts = td_acts[1].reshape((1, 1000))

        word_probs, word_selected = agents.get_sender_word_probs(target_acts, distractor_acts)

        reordering = np.array([0,1])
        random.shuffle(reordering)
        target = np.where(reordering==0)[0]

        img_array = [target_acts, distractor_acts]
        im1_acts, im2_acts = [img_array[reordering[i]] for i, img in enumerate(img_array)]

        receiver_probs, image_selected = agents.get_receiver_selection(word_selected, im1_acts, im2_acts)

        reward = 0.0
        if target == image_selected:
            reward = 1.0

        shuffled_acts = np.concatenate([im1_acts, im2_acts])

        batch.append(Game(shuffled_acts, target_acts, distractor_acts,
                          word_probs, receiver_probs, target, word_selected, image_selected, reward))

        if (i+1) % mini_batch_size == 0:
            print('updating the agent weights')

        print(target_class, reward)
        tot_reward += reward
        #selection = 0


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', required=True)
    args = parser.parse_args()
    conf = args.conf

    with open(conf) as g:
        config = yaml.load(g)

    run_game(config)

if __name__ == '__main__':
    main()
