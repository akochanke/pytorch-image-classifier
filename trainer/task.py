'''
Pytorch module to create image classifier

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
    '''Function to commence training job
    '''

    # create output folder
    model_folder = os.path.join(export_folder, 'training_' + timestamp)

    if os.path.isdir(model_folder):
        shutil.rmtree(model_folder)
    os.mkdir(model_folder)
    LOGGER.info('Saving artifacts to {}.'.format(model_folder))

    dataloaders, _, dataset_sizes = load_data(input_folder)

    model = init_model(model_type=model_type, output_folder=model_folder,
                       timestamp=timestamp)

    criterion = set_criterion()
    optimizer = set_optimizer(model)
    scheduler = set_scheduler(optimizer)

    model, history = train_model(dataloaders, dataset_sizes, model, criterion,
                                 optimizer, scheduler, num_epochs=25)

    export_model(model, model_type, history, model_folder)

def apply_model(input_folder, model_path):
    '''Function to apply model to dataset

    Parameters:
        model_path (str): location of previously trained model

    '''

    # load saved model
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

    df = get_images(input_folder)

    accuracies = []
    run = 0

    LOGGER.info('\n\n')
    for data_dict in cv_gen(df):
        run += 1
        LOGGER.info('Crossvalidation run {}'.format(run))

        dataloaders, class_names, dataset_sizes = load_data_cv(data_dict)

        model = init_model(model_type=model_type)

        criterion = set_criterion()
        optimizer = set_optimizer(model)
        scheduler = set_scheduler(optimizer)

        model, _ = train_model(dataloaders, dataset_sizes, model, criterion,
                               optimizer, scheduler, num_epochs=25)
        
        run_acc = evaluate_model(dataloaders, dataset_sizes, model)
        accuracies.append(round(float(run_acc), 2))

        LOGGER.info('Finished CV run. Accuracies: {}\n\n'.format(accuracies))

    LOGGER.info('Average accuracy: {:.2f}+={:.2f}\nAll values: {}'.format(
        np.mean(accuracies), np.std(accuracies), accuracies))


def main(job, input_folder='', export_folder='', model_type= '',
         model_path=''):
    '''Main function

    Parameters:
        input_folder (str): location of dataset
        export_folder (str): location of model artifacts

    '''

    # job id
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Folder with data.',
                        default='')
    parser.add_argument('--export', help='Store model exports.',
                        default='')
    parser.add_argument('--model', help='Model location for interence.',
                        default='')
    parser.add_argument('--job', help='Set to train or eval.',
                        default='train')
    parser.add_argument('--model_type', help='Set type of model to train.',
                        default='simplecnn')
    args = parser.parse_args()

    INPUT = args.input
    EXPORT = args.export
    MODEL = args.model
    JOB = args.job
    MODEL_TYPE = args.model_type

    main(JOB, INPUT, EXPORT, MODEL_TYPE, MODEL)
