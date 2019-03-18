import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import random


def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img.astype(np.float32)


class Environment:
    def __init__(self, data_dir, img_dirs, num_classes):
        self.img_dirs = img_dirs
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.word = None
        self.target = None
        self.distractor = None
        self.target_class = None

    def get_images(self):
        im1_class, im2_class = np.random.choice(list(range(self.num_classes)), 2, replace=False)
        #im1_class, im2_class = [0, 1]
        im1_dir, im2_dir = self.img_dirs[im1_class], self.img_dirs[im2_class]

        ## Temp var for sender training, remove later
        self.target_class = im1_dir

        im1_path, im2_path = os.path.join(self.data_dir, 'images', im1_dir), os.path.join(self.data_dir, 'images', im2_dir)

        # select random image in dirs
        im1_files, im2_files = os.listdir(im1_path), os.listdir(im2_path)
        im1_files = [i for i in im1_files if i.endswith('.jpg')]
        im2_files = [i for i in im2_files if i.endswith('.jpg')]

        im1 = np.random.choice(len(im1_files), 1)[0]
        im2 = np.random.choice(len(im2_files), 1)[0]

        target_file = os.path.join(im1_path, im1_files[im1])
        distractor_file = os.path.join(im2_path, im2_files[im2])

        # Load selected image
        self.target = load_image(target_file)
        self.distractor = load_image(distractor_file)

        return (self.target, self.distractor)


    def send_word(self, comm_word):
        self.word = comm_word

    def get_word(self):
        return self.word

    def get_round_score(self):
        return 0

    def reset(self):
        self.target = None
        self.distractor = None
        self.word = None
