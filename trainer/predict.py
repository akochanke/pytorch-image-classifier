'''
Module to do inference with a previously trained model.

'''

# imports
import time
import logging

# 3rd party
import torch


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# functions
def evaluate_model(dataloaders, dataset_sizes, model):
    '''Function to apply model to data set.

    Parameters:
        dataloaders (dict): contains the dataloader object for 'test'
        dataset_sizes (dict): sizes of dataset; here 'test'
        model (nn.Module): Pytorch model instance

    '''

    # choose hardware
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # stopwatch
    since = time.time()

    # unset dropout and batchnorm layers
    model.eval()

    running_corrects = 0

    # iterate the data set
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # model predictions
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        # count correct answers
        running_corrects += torch.sum(preds == labels.data)

    # average accuracy
    test_acc = running_corrects.double() / dataset_sizes['test']

    time_elapsed = time.time() - since
    LOGGER.info('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    LOGGER.info('Average accuracy: {:4f}'.format(test_acc))

    return test_acc
