import os
import numpy as np
import cv2
from scipy.io import loadmat

class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_paths(test_path, image_dir_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            src_full_path = os.path.join(src_path, line)
            if os.path.isfile(src_full_path):
                yield src_full_path

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        img, path,seed = self.load_data(self.training_path_infos, indices, augment=True)
        return img, path, seed
    

    def load_testing_data(self, indices):
        img, path,seed = self.load_data(self.testing_path_infos, indices)
        return img, path, seed
        

    # test ok
    def load_data(self, path_infos, indices, augment=False):
        seed = []
        path = []
        mini_batch_size = len(indices)
        in_channels = 1
        if augment:
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                path.append(path_infos[index])
                seed.append(int(path[i].split('/')[-1].split('.')[0]))
                img = cv2.imread(path[i],0)
                xs[i, 0, :, :] = (img).astype(np.float32)

        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                path.append(path_infos[index])
                seed.append(int(path[i].split('_')[-1].split('.')[0]))
                img = cv2.imread(path[i],0)
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=path))

            h, w = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs[0, 0, :, :] = (img).astype(np.float32)

        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs, path, seed
    
    