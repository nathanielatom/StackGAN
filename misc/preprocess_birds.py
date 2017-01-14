from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import tensorflow as tf
# import numpy as np
import os
import pickle
from misc.utils import get_image
import scipy.misc
import pandas as pd

# from glob import glob

# TODO: 1. current label is temporary, need to change according to real label
#       2. Current, only split the data into train, need to handel train, test

LR_HR_RETIO = 4
IMSIZE = 256
LOAD_SIZE = int(IMSIZE * 76 / 64)
BIRD_DIR = os.path.expanduser('~/Documents/datasets/birds')


def _create_directory(pathname, replace_char_dict=None):
    """
    Create *pathname* directory structure.

    Parameters
    ----------
    pathname: string
        Full name of relative or absolute path to be created. 
        If specifying a directory directly make sure to include `os.sep` at the end.
    replace_char_dict: dict, optional
        Dictionary of character sequences to replace within *pathname*. Default is {":": "-"}.

    """
    if replace_char_dict is None: replace_char_dict = {":": "-"}
    if isinstance(pathname, unicode):
        for key, val in replace_char_dict.items():
            pathname.replace(key, val)
        path = os.sep.join(pathname.split(os.sep)[:-1])
        if path:
            if not os.path.isdir(path):
                os.makedirs(path)
    return pathname


def load_filenames(data_dir):
    filepath = data_dir + 'filenames.pickle'
    with open(filepath, 'rb') as f:
        filenames = pickle.load(f)
    print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
    return filenames


def load_bbox(data_dir):
    bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
    df_bounding_boxes = pd.read_csv(bbox_path,
                                    delim_whitespace=True,
                                    header=None).astype(int)
    #
    filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
    df_filenames = pd.read_csv(filepath, delim_whitespace=True, header=None)
    filenames = df_filenames[1].tolist()
    print('Total filenames: ', len(filenames), filenames[0])
    #
    filename_bbox = {img_file[:-4]: [] for img_file in filenames}
    numImgs = len(filenames)
    for i in xrange(0, numImgs):
        # bbox = [x-left, y-top, width, height]
        bbox = df_bounding_boxes.iloc[i][1:].tolist()

        key = filenames[i][:-4]
        filename_bbox[key] = bbox
    #
    return filename_bbox


def save_data_list(inpath, outpath, filenames, filename_bbox):
    hr_images = []
    lr_images = []
    lr_size = int(LOAD_SIZE / LR_HR_RETIO)
    cnt = 0
    for key in filenames:
        bbox = filename_bbox[key]
        f_name = '%s/CUB_200_2011/images/%s.jpg' % (inpath, key)
        img = get_image(f_name, LOAD_SIZE, is_crop=True, bbox=bbox)
        img = img.astype('uint8')
        hr_images.append(img)
        lr_img = scipy.misc.imresize(img, [lr_size, lr_size], 'bicubic')
        lr_images.append(lr_img)
        cnt += 1
        if cnt % 100 == 0:
            print('Load %d......' % cnt)
    #
    print('images', len(hr_images), hr_images[0].shape, lr_images[0].shape)
    #
    outfile = outpath + str(LOAD_SIZE) + 'images.pickle'
    with open(_create_directory(outfile), 'wb') as f_out:
        pickle.dump(hr_images, f_out)
        print('save to: ', outfile)
    #
    outfile = outpath + str(lr_size) + 'images.pickle'
    with open(_create_directory(outfile), 'wb') as f_out:
        pickle.dump(lr_images, f_out)
        print('save to: ', outfile)


def split_train_test_data(filename, filename_bbox):
    all_filenames = filename_bbox.keys()
    train_filenames = []
    test_filenames = []
    with open(filename) as f:
        for line in f.readlines():
            imagindex, ttype = line.split(' ')
            if int(ttype[0]):
                test_filenames.append(all_filenames[int(imagindex) - 1])
            else:
                train_filenames.append(all_filenames[int(imagindex) - 1])
    return train_filenames, test_filenames


def convert_birds_dataset_pickle(inpath):
    # Load dictionary between image filename to its bbox
    filename_bbox = load_bbox(inpath)
    train_filenames, test_filenames = split_train_test_data(os.path.join(inpath, 'CUB_200_2011/train_test_split.txt'), filename_bbox)

    # ## For Train data
    train_dir = os.path.join(inpath, 'StackGan/train/')
    save_data_list(inpath, train_dir, train_filenames, filename_bbox)
    # ## For Test data
    test_dir = os.path.join(inpath, 'StackGan/test/')
    save_data_list(inpath, test_dir, test_filenames, filename_bbox)


if __name__ == '__main__':
    convert_birds_dataset_pickle(BIRD_DIR)
