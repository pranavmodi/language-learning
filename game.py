import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from agent_tf2 import Agents
import env
#import skimage
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

    #tf.reset_default_graph()
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

    # intermediate_layer_model = Model(inputs=model.input,
    #                                  outputs=model.get_layer(layer_name).output)

    #image_pl = tf.placeholder("float32", [1, 224, 224, 3])
    # image_pl = np.ones([1, 224, 224, 3], dtype='double')
    # vgg.build(image_pl)

    
    # if load_model==True:
    #     saver.restore(sess, load_path)
    #     print("Restored model from : ", load_path)
    # else:
    #     sess.run(tf.global_variables_initializer())
    #     print('Initialized the model')

    batch = []
    Game = namedtuple("Game", ["im_acts", "target_acts", "distractor_acts", "word_probs", "image_probs", "target", "word", "selection", "reward"])
    tot_reward = 0
    for i in range(iterations):

        print("\rEpisode {}/{}".format(i, iterations), end="")
        #sys.stdout.flush()

        if i % 10 == 0:
            print('last 10 interations performance ', tot_reward)
            tot_reward = 0

        target_image, distractor_image = environ.get_images()
        target_image = target_image.reshape((1, 224, 224, 3))
        distractor_image = target_image.reshape((1, 224, 224, 3))
        
        target_class = environ.target_class

        td_images = np.vstack([target_image, distractor_image])
        td_acts = model.predict(td_images)

        target_acts = td_acts[0].reshape((1, 1000))
        distractor_acts = td_acts[1].reshape((1, 1000))

        word_probs = agents.get_sender_word_probs(target_acts, distractor_acts)
        print(word_probs)

        continue

        reordering = np.array([0,1])
        random.shuffle(reordering)
        target = np.where(reordering==0)[0]

        img_array = [target_acts, distractor_acts] 
        i1, i2 = [img_array[reordering[i]] for i, img in enumerate(img_array)]

        shuffled_acts = np.concatenate([i1, i2], axis=1)

        ## for Sender - take action in reinforcement learning terms

        reward, word, selection, word_probs, image_probs = agents.show_images(sess, shuffled_acts, target_acts, distractor_acts, target, target_class)

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

    run_game(config)

if __name__ == '__main__':
    main()
