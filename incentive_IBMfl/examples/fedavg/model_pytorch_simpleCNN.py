import os 
import torch
from torch import nn


def get_hyperparams():
    local_params = {
        'training': {
            'epochs': 3,
            'lr': 1
        },
    }
    
    return local_params


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0):

    if is_agg:
        return None
    # model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16

    #         nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8

    #         nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(),
    #         nn.MaxPool2d(2, 2),  # output: 256 x 4 x 4

    #         nn.Flatten(),
    #         nn.Linear(256 * 4 * 4, 1024),
    #         nn.ReLU(),
    #         nn.Linear(1024, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 10),
    #         nn.LogSoftmax(dim=1))
    model = nn.Sequential(nn.Conv2d(3, 32, 3, 1),
                        nn.ReLU(),
                        nn.Conv2d(32, 64, 3, 1),
                        nn.ReLU(),
                        nn.MaxPool2d(2, 2),
                        nn.Dropout2d(p=0.25),
                        nn.Flatten(),
                        nn.Linear(12544, 128),
                        nn.ReLU(),
                        nn.Dropout2d(p=0.5),
                        nn.Linear(128, 10),
                        nn.LogSoftmax(dim=1)
                        )
    
    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)

    # Save model
    fname = os.path.join(folder_configs, 'pytorch_sequence.pt')
    torch.save(model, fname)

    # Specify an optimizer class as optim.<optimizer> 
    # The entire expression should be of type string
    # e.g., optimizer = 'optim.SGD'
    optimizer = 'optim.Adadelta'
    # Specify a loss criterion as nn.<loss-criterion>
    # The entire expression should be of type string
    # e.g., criterion = 'nn.NLLLoss'
    criterion = 'nn.NLLLoss'
    spec = {
        'model_name': 'pytorch-nn',
        'model_definition': fname,
        'optimizer': optimizer,
        'loss_criterion': criterion,
    }
    model = {
        'name': 'PytorchFLModel',
        'path': 'ibmfl.model.pytorch_fl_model',
        'spec': spec,
    }
    return model
