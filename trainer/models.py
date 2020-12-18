'''
Module to collect used models

'''

# imports
import os
import shutil
import logging
import datetime

# 3rd party
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# set logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(module)s:%(funcName)s :: %(message)s',
    datefmt='%H:%M:%S')
LOGGER = logging.getLogger(__name__)
LOGGER.level = logging.INFO


# classes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5,
                               stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5,
                               stride=1, padding=2)
        self.fc1 = nn.Linear(in_features=1024, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = self.fc3(x)
        return output
    
    def num_flat_features(self, x):
        '''Count dimensionality of layer
        '''

        size = x.size()[1:] # all dims without batch
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# functions
def vis_history(history, location):
    '''Function to visualize trianing history
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

def export_model(model, history, export_folder):
    '''Function to export model artifacts

    Parameters:
        export_folder (str): where model files will be saved

    '''

    # timestamp suffix for separation
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y%m%d%H%M%S')
    model_folder = os.path.join(export_folder, 'simplecnn_' + timestamp)

    if os.path.isdir(model_folder):
        shutil.rmtree(model_folder)
    os.mkdir(model_folder)

    # save model binary
    torch.save(model.state_dict(), os.path.join(model_folder, 'simplecnn.pth'))

    # save model architecture
    summary_path = os.path.join(model_folder, 'model_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary(model, input_size=(3, 256, 256)))
    
    # save train history
    vis_history(history, model_folder)
