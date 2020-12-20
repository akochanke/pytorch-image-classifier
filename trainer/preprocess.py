'''
Module part to collect image preprocessing steps

'''

# imports
import os
import logging

# 3rd party
import torch
from torchvision import datasets, transforms


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# input transformations
TRAINING = transforms.Compose([
    #transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(p=0.25),
    transforms.RandomVerticalFlip(p=0.25),
    transforms.RandomApply(torch.nn.ModuleList([
        transforms.ColorJitter(),
        transforms.GaussianBlur(kernel_size=5)
    ]), p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
EVALUATION = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TEST = EVALUATION


# functions
def load_data(input_folder, batch_size=4):
    '''Function to initiate dataloaders for Pytorch

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
    '''Function to load test set.

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
