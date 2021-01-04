'''
Pytorch module to create image classifier. Supported tasks:

- training
- model inference
- crossvalidation

'''

# imports
import os
import shutil
import datetime
import argparse
import logging

# 3rd party
import numpy as np

# module imports
from trainer.preprocess import (load_data, load_test, get_images, cv_gen,
                                load_data_cv)
from trainer.training import (train_model, set_criterion, set_optimizer,
                              set_scheduler)
from trainer.predict import evaluate_model
from trainer.models import (export_model, init_model, load_model)


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# functions
def train(input_folder, export_folder, model_type, timestamp):
    '''Function to commence training job.

    Parameters:
        input_folder (str): location of dataset
        export_folder (str): location for artifact export
        model_type (str): define model architecture
        timestamp (str): job id
    '''

    # create output folder
    model_folder = os.path.join(export_folder, 'training_' + timestamp)
    if os.path.isdir(model_folder):
        shutil.rmtree(model_folder)
    os.mkdir(model_folder)
    LOGGER.info('Saving artifacts to {}.'.format(model_folder))

    # prepare dataset via dataloader class
    dataloaders, _, dataset_sizes = load_data(input_folder)

    # initiate model instance
    model = init_model(model_type=model_type, output_folder=model_folder)

    # further training objects
    criterion = set_criterion()
    optimizer = set_optimizer(model)
    scheduler = set_scheduler(optimizer)

    # commence training
    model, history = train_model(dataloaders, dataset_sizes, model, criterion,
                                 optimizer, scheduler, num_epochs=25)

    # export artifacts
    export_model(model, model_type, history, model_folder)

def apply_model(input_folder, model_path):
    '''Function to apply previously trained model to dataset.

    Parameters:
        input_folder (str): location of dataset; expects `test` subfolder
        model_path (str): location of previously trained model

    '''

    # load saved model `*.pth`
    model = load_model(model_path)

    # prepare test set
    dataloaders, _, dataset_sizes = load_test(input_folder)

    # apply model to test set
    evaluate_model(dataloaders, dataset_sizes, model)

def crossvalidation(input_folder, model_type):
    '''Function to run a crossvalidation on the full dataset

    Parameters:
        input_folder (str): location of dataset
        model_type (str): set used model type

    '''

    # collect all images in dataframe
    df = get_images(input_folder)

    accuracies = []
    run = 0

    # CV loop
    print('\n\n')
    for data_dict in cv_gen(df):
        run += 1
        LOGGER.info('Crossvalidation run {}'.format(run))

        # create dataloader object from cv generator
        dataloaders, _, dataset_sizes = load_data_cv(data_dict)

        # initialize model object
        model = init_model(model_type=model_type)

        # further training objects
        criterion = set_criterion()
        optimizer = set_optimizer(model)
        scheduler = set_scheduler(optimizer)

        # commence training
        model, _ = train_model(dataloaders, dataset_sizes, model, criterion,
                               optimizer, scheduler, num_epochs=25)

        # evaluate model on test subset
        run_acc = evaluate_model(dataloaders, dataset_sizes, model)

        # collect accuracies during cv
        accuracies.append(round(float(run_acc), 2))

        LOGGER.info('Finished CV run. Accuracies: {}'.format(accuracies))
        print('\n\n')

    LOGGER.info('Average accuracy: {:.2f}+={:.2f}\nAll values: {}'.format(
        np.mean(accuracies), np.std(accuracies), accuracies))

def main(job, input_folder='', export_folder='', model_type= '',
         model_path=''):
    '''Main function to handle jobs.

    Parameters:
        job (str): define type of job training/evaluation/crossvalidation
        input_folder (str): location of dataset
        export_folder (str): location of model artifacts
        model_type (str): choose model architecture simplecnn/resnet18
        model_path (str): location of model artifacts for inference

    '''

    # generating job id
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')

    if job == 'training':
        train(input_folder, export_folder, model_type, timestamp)
    elif job == 'evaluation':
        apply_model(input_folder, model_path)
    elif job =='crossvalidation':
        crossvalidation(input_folder, model_type)
    else:
        LOGGER.error('Job type {} not recognized!'.format(job))
        raise NotImplementedError


if __name__ == '__main__':

    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Set folder for image data.',
                        default='')
    parser.add_argument('--export', help='Set folder for artifact exports.',
                        default='')
    parser.add_argument('--model', help='Model location for inference.',
                        default='')
    parser.add_argument('--job',
                        help='Set to training, evaluation or crossvalidation.',
                        default='train')
    parser.add_argument('--model_type',
                        help='Set type of model (simplecnn/resnet18).',
                        default='simplecnn')
    args = parser.parse_args()

    INPUT = args.input
    EXPORT = args.export
    MODEL = args.model
    JOB = args.job
    MODEL_TYPE = args.model_type

    LOGGER.info('Parsed arguments: {}'.format(args))

    # run module
    main(JOB, INPUT, EXPORT, MODEL_TYPE, MODEL)
