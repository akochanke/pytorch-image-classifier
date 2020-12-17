'''
Pytorch module to create image classifier

'''

# imports
import argparse
import logging


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# functions
def main(input_folder, export_folder):
    '''Main function

    Parameters:
        input_folder (str): location of dataset
        export_folder (str): location of model artifacts

    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='Folder with data. Expects train/val.',
                        default='data/challenge_256')
    parser.add_argument('--export', help='Store model exports.',
                        default='artifacts')
    args = parser.parse_args()

    INPUT = args.input
    EXPORT = args.export

    main(INPUT, EXPORT)
