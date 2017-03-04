import os
import numpy as np
import tensorflow as tf
import sys
sys.path.append('tensorflow-vgg/')
import utils
import vgg16

## This class contains the Sender Agent which recieves the activations of the target image and distractor image, maps them into vocabulary words and sends it the reciever
class SenderAgent:

    def __init__(self, embedding_dim = 2):

        self.vgg = vgg16.Vgg16()

    def show_images(self, target, distractor):
        images = tf.placeholder("float", [2, 224, 224, 3])

        batch1 = target.reshape((1, 224, 224, 3))
        batch2 = distractor.reshape((1, 224, 224, 3))

        print(batch1.shape)
        batch = np.concatenate((batch1, batch2), 0)

        with tf.Session(
                config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
            images = tf.placeholder("float", [2, 224, 224, 3])
            feed_dict = {images: batch}

            vgg = vgg16.Vgg16()
            with tf.name_scope("content_vgg"):
                vgg.build(images)
                prob = sess.run(vgg.prob, feed_dict=feed_dict)

        comm_word = ''
        return comm_word



class RecieverAgent:
    def __init__(self, activation_shape=[1000, 1]):
        self.something = []

        self.vgg = vgg16.Vgg16()

    def recieve(comm_word, image1, image2):
        return 0
