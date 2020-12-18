'''
Module part to control the trianing

'''

# imports
import os
import time
import copy
import logging

# 3rd party
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# function
def set_criterion():
    '''function to define criterion for training

    Return:
        instance of loss class CrossEntropyLoss

    '''

    # return instance of loss class
    return nn.CrossEntropyLoss()

def set_optimizer(model, optimizer='sdg'):
    '''Function to define used optimizer during training.
    Supported: SDG, Adam

    Parameters:
        model ():
        optimizer (str): name of optimizer

    Return:
        opt (optim.XXX): instance of optimizer class

    '''

    assert optimizer in ['sdg', 'adam']

    if optimizer == 'sdg':
        opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    elif optimizer == 'adam':
        opt = optim.Adam(model.parameters(), lr=0.001)
    
    return opt

def set_scheduler(optimizer):
    '''Function to define the scheduler during training

    Return:
        scheduler instance of class StepLR

    '''

    return lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def train_model(dataloaders, dataset_sizes, model, criterion, optimizer,
                scheduler, num_epochs=25):
    '''Function to train a model

    Parameters:
        model

    Return:
        model

    '''

    # choose hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # dictionary to save train history
    history = {'training': {'loss': [], 'acc': []},
               'validation': {'loss': [], 'acc': []}}

    for epoch in range(num_epochs):
        LOGGER.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        LOGGER.info('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    print(outputs, labels)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'training':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            LOGGER.info('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # save history
            history[phase]['loss'].append(epoch_loss)
            history[phase]['acc'].append(epoch_acc)

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    LOGGER.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    LOGGER.info('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history
