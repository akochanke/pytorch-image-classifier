'''
Module to define model classes and model functionality.

'''

# imports
import os
import logging

# 3rd party
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# classes
class SimpleCNN(nn.Module):
    '''Simple model class based von conv2d layers and a fully connected head.
    '''

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=5, stride=1, padding=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=5, stride=1, padding=2)
        #self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
        #                       kernel_size=5, stride=1, padding=2)
        #self.fc1 = nn.Linear(in_features=128*16*16, out_features=64)
        #self.fc2 = nn.Linear(in_features=128*16*16, out_features=64)
        self.fc3 = nn.Linear(in_features=128*16*16, out_features=4)

    def forward(self, x):
        '''Define architecture in functional style.
        '''

        # conv-net
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = self.maxpool(F.relu(self.conv3(x)))
        x = self.maxpool(F.relu(self.conv4(x)))
        #x = self.maxpool(F.relu(self.conv5(x)))

        # fc-net
        x = x.view(-1, 128*16*16)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        output = self.fc3(x)

        return output

    @staticmethod
    def num_flat_features(x):
        '''Count dimensionality of layer.
        '''

        size = x.size()[1:] # all dims without batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# functions
def init_model(model_type='simplecnn', output_folder=''):
    '''Function to initiate model object and save architecture to file.

    Parameters:
        model_type (str): define model type
        output_folder (str): location of model architecture file

    Return:
        model (torch.model): model instance

    '''

    LOGGER.info('Initiating model type {}...'.format(model_type))

    # choose model type
    if model_type == 'simplecnn':
        model = SimpleCNN()

    elif model_type == 'resnet18':
        model = models.resnet18(pretrained=True)

        # freeze layers
        for idx, param in enumerate(model.parameters()):
            if idx <= 40:
                param.requires_grad = False

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 4)

    else:
        LOGGER.error('Unknown model type {}!'.format(model_type))
        raise NotImplementedError

    # safe model architecture to file
    if output_folder:
        save_architecture(model_type, model, output_folder)

    return model

def load_model(model_path):
    '''Function to load previously trained model from `*.pth`. Note that file
    with full model export is expected, i.e. via `torch.save()`.

    Parameters:
        model_folder (str): folder with model file

    Return:
        model (nn.Module): pytorch model instance

    '''

    # get model file name
    model_file = [f for f in os.listdir(model_path) if f.endswith('.pth')]
    assert len(model_file) == 1, 'Multiple model files not supported'
    model_path = os.path.join(model_path, model_file[0])

    # load model from path
    LOGGER.info('Loading model from {}...'.format(model_path))
    model = torch.load(model_path)

    return model

def save_architecture(model_type, model, export_folder,
                      input_layer=(3, 256, 256)):
    '''Function to save model architecture to txt file.

    Parameters:
        model_type (str): name of model type
        model (nn.Module): pytorch model instance
        export_folder (str): folder for file export
        input_layer (tuple): input dimenions of model for forward propagation

    '''

    summary_file = os.path.join(export_folder, model_type + '_summary.txt')

    model_summary = summary(model, input_layer)

    with open(summary_file, 'w') as f:
        f.write(str(model_summary)) # type of model_summary is ModelStatistics

def vis_history(history, location):
    '''Function to visualize trianing history

    Parameters:
        history (dict): training history (loss/accuracy)
        location (str): path to plot location

    '''

    plt.figure(figsize=(10, 10), dpi=150)
    plt.title('Model loss and accuracy')

    for phase, phase_data in history.items():
        for metric, metric_data in phase_data.items():
            plt.plot(list(range(1, len(metric_data)+1)), metric_data,
                     label='_'.join([phase, metric]))

    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.grid(True)
    plt.savefig(os.path.join(location, 'history.pdf'))
    plt.close()

def export_model(model, model_type, history, export_folder):
    '''Function to export model artifacts.

    Parameters:
        model (nn.Module): Pytorch model instance
        model_type (str): name of model type
        history (dict): training history (loss/accuracy)
        export_folder (str): where model files will be saved

    '''

    # save model binary (weights + architecture)
    torch.save(model,
               os.path.join(export_folder, model_type + '.pth'))

    # save train history to plot
    vis_history(history, export_folder)
