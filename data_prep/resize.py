'''
Program to carry out data preparation tasks

'''

# imports
import os
import shutil
import argparse
import logging
from multiprocessing import Pool

# 3rd party
from PIL import Image


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# function
def resize_image(input_path, output_path, resolution):
    '''Function to resize single image

    Parameters:
        input_path (str): location of image
        output_path (str): output location of image
        resolution (int): output resolution of image

    '''

    LOGGER.info('Processing {}/{}...'.format(input_path, output_path))
    img = Image.open(input_path)
    img_resized = img.resize((resolution, resolution))
    img_resized.save(output_path, 'TIFF')

def io_list(input_path, output_path, resolution):
    '''Function to prepare an input/output list

    Parameters:
        input_path (str): location of images
        output_path (str): location of images after transformation

    Return:
        job_list (list): each element is a tuple (input, output, resolution)

    '''

    LOGGER.info('Creating task list...')

    job_list = []

    #root, folder = os.path.split(input_path)
    classes = os.listdir(input_path)

    for cls in classes:
        image_list = os.listdir(os.path.join(input_path, cls))
        for img in image_list:
            job_list.append(
                (os.path.join(input_path, cls, img),
                 os.path.join(output_path, cls, img),
                 resolution)
            )

    LOGGER.info('{} files will be processed.'.format(len(job_list)))

    return job_list

def prep_folder(input_folder, output_folder):
    '''Function to create new dataset folder; preserves structure

    Parameters:
        input_folder (str): original dataset
        output_folder (str): transformed dataset

    '''

    classes = os.listdir(input_folder)

    if os.path.isdir(output_folder):
        LOGGER.info('Output folder {} already existed; deleting it...' \
                    .format(output_folder))
        shutil.rmtree(output_folder)

    LOGGER.info('Creating {}...'.format(output_folder))
    os.mkdir(output_folder)
    for cls in classes:
        os.mkdir(os.path.join(output_folder, cls))

def main(input_folder, output_folder, resolution):
    '''Main function

    Parameters:
        input_folder (str): original dataset location
        output_folder (str): dataset after transformation
        resolution (int): new image resolution

    '''

    prep_folder(input_folder, output_folder)

    queue = io_list(input_folder, output_folder, resolution)

    with Pool(4) as p:
        p.starmap(resize_image, queue)


if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Folder with data',
                        default='data/challenge')
    parser.add_argument('--output_folder', help='Store output data',
                        default='data/challenge_256')
    parser.add_argument('--resolution', help='Output image resolution',
                        default=256)
    args = parser.parse_args()

    INPUT_FOLDER = args.input_folder
    OUTPUT_FOLDER = args.output_folder
    RESOLUTION = args.resolution

    main(INPUT_FOLDER, OUTPUT_FOLDER, RESOLUTION)
