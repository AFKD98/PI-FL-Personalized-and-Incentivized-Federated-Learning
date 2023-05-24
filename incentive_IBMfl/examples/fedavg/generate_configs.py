import os
import numpy as np
from importlib import import_module

import examples.datahandlers as datahandlers


def get_fusion_config():
    fusion = {
        'name': 'TierFedAvgFusionHandler',
        'path': 'ibmfl.aggregator.fusion.tier_fedavg_fusion_handler'
    }
    return fusion


def get_local_training_config(configs_folder=None):
    local_training_handler = {
        'name': 'TierFedAvgLocalTrainingHandler',
        'path': 'ibmfl.party.training.tier_fedavg_local_training_handler'
    }
    return local_training_handler


def get_hyperparams(model):
    hyperparams = {
        'global': {
                'tiers': 5,
                'tokens': 500,
                'rounds': 800,
                'termination_accuracy': 0.9,
                'max_timeout': 10000,
                'token_to_pay' : 1,
                'parties_selected_per_tier' : 4,
                'random_parties_selected_per_tier': 1,
                'pre_training_rounds': 0,
                'select_random': False,
            }
    }
    current_module = globals().get('__package__')
    
    model_module = import_module('{}.model_{}'.format(current_module, model))
    local_params_method = getattr(model_module, 'get_hyperparams')

    local_params = local_params_method()
    hyperparams['local'] = local_params
    
    return hyperparams


def get_data_handler_config(party_id, dataset, folder_data, is_agg=False, model='keras'):

    SUPPORTED_DATASETS = ['mnist', 'custom_dataset', 'cifar10', 'emnist']
    if dataset in SUPPORTED_DATASETS:
        if model not in 'keras':
            dataset = dataset + "_" + model

        data = datahandlers.get_datahandler_config(
            dataset, folder_data, party_id, is_agg)
    else:
        raise Exception(
            "The dataset {} is a wrong combination for fusion/model".format(dataset))
    return data


def get_model_config(folder_configs, dataset, is_agg=False, party_id=0, model='keras'):
    SUPPORTED_MODELS = ['keras', 'pytorch', 'tf', 'sklearn']

    if model not in SUPPORTED_MODELS:
        raise Exception("Invalid model config for this fusion algorithm")

    current_module = globals().get('__package__')
    
    model_module = import_module('{}.model_{}'.format(current_module, model))
    method = getattr(model_module, 'get_model_config')

    return method(folder_configs, dataset, is_agg=is_agg, party_id=0)


