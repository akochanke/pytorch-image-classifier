'''
Program to carry out data preparation tasks:

- creates folder structure train/eval/test
- images are shuffled
- images can be resized

'''

# imports
import os
import shutil
import argparse
import logging
import random
from itertools import zip_longest
from multiprocessing import Pool

# 3rd party
from PIL import Image


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# parameters
DATASPLIT = {'training': 0.8, 'validation': 0.1, 'test': 0.1}
SEED = 42 # for reproducability
random.seed(SEED)


# functions
def resize_image(input_path, output_path, resolution, file_extension='PNG'):
    '''Function to resize single image

    Parameters:
        input_path (str): location of single image
        output_path (str): output location of image
        resolution (int): output resolution of image
        file_extension (str): file type for pillow

    '''

    LOGGER.info('Processing {} to {}...'.format(input_path, output_path))
    img = Image.open(input_path)
    if resolution:
        img = img.resize((resolution, resolution))
    img.save(output_path, file_extension)

def shuffle_split(amount=100):
    '''Function to generate a shuffled list for train/eval/test split. Need
    to extend this function if classes are imbalanced.

    Parameters:
        amount (int): amount of files per class

    '''

    groups = {'training': ['training'],
              'validation': ['validation'],
              'test': ['test']}

    # create list with one of train/eval/test per example
    data_splitting = []
    for g, frac in DATASPLIT.items():
        data_splitting += int(amount*frac) * groups[g]

    # shuffle the list; inplace
    random.shuffle(data_splitting)

    return data_splitting

def io_list(input_path, output_path, resolution, split, file_extension='.png'):
    '''Function to prepare an input/output list. This list can easily be parsed
    to multiprocessing.

    Parameters:
        input_path (str): location of images
        output_path (str): location of images after transformation
        resolution (int): resolution of transformed image (squared)
        split (bool): set to generate train/eval/test split
        file_extension (str): new file extension

    Return:
        job_list (list): each element is a tuple (input, output, resolution)

    '''

    LOGGER.info('Creating task list...')

    job_list = []

    classes = os.listdir(input_path)

    # randomly assign examples to train/eval/test
    data_splitting = shuffle_split()

    for cls in classes:
        image_list = os.listdir(os.path.join(input_path, cls))

        if len(image_list) != len(data_splitting):
            LOGGER.warning(
                'Amount of images and dataset splitting does not match!')

        for img, spl in zip_longest(image_list, data_splitting,
                                    fillvalue='training'):
            new_img = os.path.splitext(img)[0] + file_extension
            origin = os.path.join(input_path, cls, img)

            if split:
                destination = os.path.join(output_path, spl, cls, new_img)
            else:
                destination = os.path.join(output_path, cls, new_img)

            job_list.append((origin, destination, resolution))

    LOGGER.info('{} files will be processed.'.format(len(job_list)))

    return job_list

def prep_folder(input_folder, output_folder, split):
    '''Function to create new dataset folder; preserves class structure;
    already existing folders are overwritten.

    Parameters:
        input_folder (str): original dataset
        output_folder (str): transformed dataset
        split (bool): set to generate train/eval/test split

    '''

    classes = os.listdir(input_folder)

    if os.path.isdir(output_folder):
        LOGGER.info('Output folder {} already existed; deleting it...' \
                    .format(output_folder))
        shutil.rmtree(output_folder)

    LOGGER.info('Creating {}...'.format(output_folder))
    os.mkdir(output_folder)

    if split:
        for spl in DATASPLIT:
            split_folder = os.path.join(output_folder, spl)
            os.mkdir(split_folder)
            for cls in classes:
                os.mkdir(os.path.join(split_folder, cls))
    else:
        for cls in classes:
            os.mkdir(os.path.join(output_folder, cls))

def main(input_folder, output_folder, resolution, split):
    '''Main function for data preparation.

    Parameters:
        input_folder (str): original dataset location
        output_folder (str): dataset location after transformation
        resolution (int): new image resolution (squared)
        split (bool): set to generate train/eval/test split

    '''

    # create folder structure
    prep_folder(input_folder, output_folder, split)

    # generate transformation queue
    queue = io_list(input_folder, output_folder, resolution, split)

    # run queue via multiprocessing
    with Pool(4) as p:
        p.starmap(resize_image, queue) # starmap to unwrap queue


if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Folder with data',
                        default='data/challenge')
    parser.add_argument('--output_folder', help='Store output data',
                        default='data/challenge_png')
    parser.add_argument('--resolution', help='Output image resolution',
                        type=int, default=0)
    parser.add_argument('--split', help='Set true for train/eval/test split.',
                        action='store_true')
    args = parser.parse_args()

    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    RESOLUTION = args.resolution
    SPLIT = args.split

    LOGGER.info('Arguments: {}'.format(args))

    # run module
    main(INPUT_FOLDER, OUTPUT_FOLDER, RESOLUTION, SPLIT)
