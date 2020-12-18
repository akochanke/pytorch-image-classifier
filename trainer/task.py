'''
Pytorch module to create image classifier

'''

# imports
import argparse
import logging

# 3rd party
import torchsummary as summary

# module imports
from trainer.preprocess import load_data
from trainer.training import (train_model, set_criterion, set_optimizer,
                              set_scheduler)
from trainer.models import (SimpleCNN, export_model)


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# functions
def train(input_folder, export_folder):
    '''Function to commence training job
    '''

    dataloaders, class_names, dataset_sizes = load_data(input_folder)

    model = SimpleCNN()
    #LOGGER.info('Model architecture\n{}'.format(summary(model, (3,256,256))))

    criterion = set_criterion()
    optimizer = set_optimizer(model)
    scheduler = set_scheduler(optimizer)

    model, history = train_model(dataloaders, dataset_sizes, model, criterion,
                                 optimizer, scheduler, num_epochs=1)

    export_model(model, history, export_folder)

def apply_model(model_path):
    '''Function to apply model to dataset

    Parameters:
        model_path (str): location of previously trained model

    '''

    pass

def main(job, input_folder='', export_folder='', model_path=''):
    '''Main function

    Parameters:
        input_folder (str): location of dataset
        export_folder (str): location of model artifacts

    '''

    assert job in ['train', 'eval'], 'Unknown job type!'

    if job == 'train':
        train(input_folder, export_folder)
    elif job == 'eval':
        apply_model(model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Folder with data. Expects train/val.',
                        default='data/challenge_256')
    parser.add_argument('--export', help='Store model exports.',
                        default='artifacts')
    parser.add_argument('--model', help='Location of model file.',
                        default='')
    parser.add_argument('--job', help='Set to train or eval.',
                        default='train')
    args = parser.parse_args()

    INPUT = args.input
    EXPORT = args.export
    MODEL = args.model
    JOB = args.job

    main(JOB, INPUT, EXPORT, MODEL)
