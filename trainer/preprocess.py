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


# functions
def load_data(input_folder):
    '''Function to initiate dataloaders for Pytorch

    Parameters:
        input_folder (str): data location; note that train/val is expected

    Return:
        dataloaders (dict): contains Dataloader objects for training and
        validation
        class_names (list): list of class names
        dataset_size (dict): amount of images in 'training' and 'evaluation'

    '''

    # define augmentation and preprocessing; images seem to be scaled to (0, 1)
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(input_folder, x),
                                              data_transforms[x])
                      for x in ['training', 'validation']
                    }

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['training', 'validation']
                }

    dataset_sizes = {x: len(image_datasets[x])
                     for x in ['training', 'validation']
                    }
    LOGGER.info('The dataset consists of {}'.format(dataset_sizes))

    class_names = image_datasets['training'].classes

    return dataloaders, class_names, dataset_sizes
