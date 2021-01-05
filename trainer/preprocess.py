'''
Module to define input pipelines

'''

# imports
import os
import logging

# 3rd party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, transforms
from sklearn.model_selection import (StratifiedKFold, train_test_split)


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# input transformations
TRAINING = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(brightness=(0.7, 1.3),
                               contrast=(0.7, 1.3),
                               saturation=(0.7, 1.3),
                               hue=(-0.1, 0.1)),
        transforms.GaussianBlur(kernel_size=9, sigma=3.)
    ]), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
EVALUATION = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TEST = EVALUATION


# classes
class DatasetFromPath(Dataset):
    '''Custom dataset class to work with a list of image paths/labels.
    '''

    def __init__(self, data_tuple, transform):
        self.image_list = data_tuple[0]
        self.labels = data_tuple[1]
        self.classes = list(set(self.labels))
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_loc = self.image_list[idx]
        img_label = int(self.labels[idx][-1]) - 1 # e.g class1 -> 0
        image = Image.open(img_loc)
        tensor_image = self.transform(image)
        return tensor_image, img_label


# functions
def load_data(input_folder, batch_size=4):
    '''Function to initiate dataloaders for training.

    Parameters:
        input_folder (str): data location; note that train/val is expected
        batch_size (int): size of mini batch

    Return:
        dataloaders (dict): contains Dataloader objects for training and
        validation
        class_names (list): list of class names
        dataset_size (dict): amount of images in 'training' and 'evaluation'

    '''

    LOGGER.info('Loading data from {}'.format(input_folder))

    # define augmentation and preprocessing; images seem to be scaled to (0, 1)
    data_transforms = {
        'training': TRAINING,
        'validation': EVALUATION,
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(input_folder, x),
                                              data_transforms[x])
                      for x in ['training', 'validation']
                    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['training', 'validation']
                }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['training', 'validation']
                    }

    LOGGER.info('The data set consists of {}'.format(dataset_sizes))

    class_names = image_datasets['training'].classes

    return dataloaders, class_names, dataset_sizes

def load_test(input_folder, batch_size=4):
    '''Function to create dataloader for test set.

    Parameters:
        input_folder (str): data location; note that train/val is expected
        batch_size (int): size of image batch

    Return:
        dataloaders (dict): contains Dataloader objects for training and
        validation
        class_names (list): list of class names
        dataset_size (dict): amount of images in 'training' and 'evaluation'

    '''

    LOGGER.info('Loading data from {}'.format(input_folder))

    data_transforms = {
        'test': TEST
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(input_folder, x),
                                            data_transforms[x])
                    for x in ['test']
                }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['test']
                }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['test']
                    }

    LOGGER.info('The test set consists of {}'.format(dataset_sizes))

    class_names = image_datasets['test'].classes

    return dataloaders, class_names, dataset_sizes

def imshow(inp, title=None):
    '''Plot image for inspection. Note the rev normalization due to input
    transformations.

    Parameters:
        inp (torchvision.utils.make_grid()): image sample
        title (str): default title

    '''

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def vis_from_dataloaders(dataloaders, class_names):
    '''Function to visualize images from dataloaders to inspect augmentation

    Parameters:
        dataloaders (dict): dictionary with dataloader instances
        class_names (list): list of class names for captions

    '''

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['training']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # visualize some images from dataloader instance
    imshow(out, title=[class_names[x] for x in classes])

def get_images(input_folder):
    '''Function to collect all image paths within a folder and store these
    together with their labels in a dataframe.

    Parameters:
        input_folder (str): path to image folder

    Return:
        df (pd.DataFrame): table of image data/labels

    '''

    LOGGER.info('Generating image list from folder {}...'.format(input_folder))

    columns = ['image_path', 'label']
    rows = []

    for cls in os.listdir(input_folder):
        label = cls #int(cls[-1]) - 1
        class_folder = os.path.join(input_folder, cls)
        img_list = [os.path.join(input_folder, cls, img_name)
                    for img_name in os.listdir(class_folder)]
        class_list = list(zip(img_list, len(img_list)*[label]))
        rows += class_list

    df = pd.DataFrame(data=rows, columns=columns)

    return df

def cv_gen(df_data):
    '''Function to create generator for crossvalidation.

    Parameters:
        df_data (pd.DataFrame): table with images and labels

    Return:
        data_dict (dict): containing image/label lists for train/eval/test

    '''

    data_dict = {}

    # stratified kfold to preserve class percentages
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # split fold again into train/eval/test
    for train, test in skf.split(df_data['image_path'], df_data['label']):
        X, y = df_data['image_path'][train], df_data['label'][train]
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=1/9, random_state=42, stratify=y)

        data_dict['training'] = X_train.tolist(), y_train.tolist()
        data_dict['validation'] = X_val.tolist(), y_val.tolist()
        data_dict['test'] = (df_data['image_path'][test].tolist(),
                             df_data['label'][test].tolist())

        yield data_dict

def load_data_cv(image_dict, batch_size=4):
    '''Function to create dataloaders from cv generator

    Parameters:
        image_dict (dict): contains image/label lists for train/eval/test
        batch_size (int): size of batches

    Return:
        dataloaders (dict): contains Dataloader objects for training,
        validation and test
        class_names (list): list of class names
        dataset_size (dict): amount of images in 'training', 'evaluation' and
        'test'

    '''

    # define augmentation and preprocessing; images seem to be scaled to (0, 1)
    data_transforms = {
        'training': TRAINING,
        'validation': EVALUATION,
        'test': TEST
    }

    # dataset object from custom class
    image_datasets = {x: DatasetFromPath(image_dict[x], data_transforms[x])
                      for x in ['training', 'validation', 'test']
                    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['training', 'validation', 'test']
                }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['training', 'validation', 'test']
                    }

    LOGGER.info('The data set consists of {}'.format(dataset_sizes))

    class_names = image_datasets['training'].classes

    return dataloaders, class_names, dataset_sizes
