#!/usr/bin/env python3

import argparse
import os
import time
import yaml
import json
import sys
from importlib import import_module

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from examples.constants import GENERATE_CONFIG_DESC, NUM_PARTIES_DESC, \
    PATH_CONFIG_DESC, CONF_PATH, MODEL_CONFIG_DESC, NEW_DESC, NAME_DESC, \
    FL_EXAMPLES, FL_CONN_TYPES, CONNECTION_TYPE_DESC, FL_MODELS, \
    FUSION_CONFIG_DESC, TASK_NAME_DESC, CONTEXT_PATH, PARTY_IP, NUM_TIERS

from examples.constants import FL_CONTEXT


def check_valid_folder_structure(p):
    """
    Checks that the folder structure is valid

    :param p: an argument parser
    :type p: argparse.ArgumentParser
    """
    for folder in FL_EXAMPLES:
        if not os.path.isfile(os.path.join("examples", folder, "README.md")) and not os.path.isfile(os.path.join(
                "examples", folder, "generate_configs.py")):
            p.error(
                "Bad folder structure: '{}' directory is missing files.".format(folder))


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_CONFIG_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--num_tiers", "-tn", help=NUM_TIERS,
                   type=int, required=False)
    p.add_argument("--dataset", "-d",
                   help="Dataset code from examples", type=str, required=True)

    p.add_argument("--data_path", "-p", help=PATH_CONFIG_DESC, required=True)
    p.add_argument("--config_path", "-conf_path", help=CONF_PATH)
    p.add_argument("--model", "-m", help=MODEL_CONFIG_DESC, choices=[os.path.basename(
        d) for d in FL_MODELS], required=False, default=None)
    p.add_argument("--fusion", "-f", help=FUSION_CONFIG_DESC ,required=True, choices=[os.path.basename(
        d) for d in FL_EXAMPLES])
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    p.add_argument("--connection", "-c", choices=[os.path.basename(
        d) for d in FL_CONN_TYPES], help=CONNECTION_TYPE_DESC, required=False, default="flask")
    p.add_argument("--task_name", "-t", help=TASK_NAME_DESC, required=False)
    p.add_argument("--context_path", "-context", help=CONTEXT_PATH)
    p.add_argument("--party_ip", "-party_ip", help=PARTY_IP)
    return p


def generate_connection_config(conn_type, party_id=0, is_party=False, task_name=None, party_ip='127.0.0.1'):
    connection = {}

    if conn_type == 'flask':
        tls_config = {
            'enable': False
        }
        connection = {
            'name': 'FlaskConnection',
            'path': 'ibmfl.connection.flask_connection',
            'sync': False
        }
        if is_party:
            connection['info'] = {
                'ip': party_ip,
                'port': 8085 + party_id
            }
        else:
            connection['info'] = {
                'ip': '192.168.0.231',
                'port': 5000
            }
        connection['info']['tls_config'] = tls_config
    if conn_type == 'pubsub':
        connection = {
            'name': 'RabbitMQConnection',
            'path': 'ibmfl.connection.rabbitmq_connection',
            'sync': True
        }
        if is_party:
            connection['info'] = {
                'credentials': f'party{party_id}.json',
                'role': 'party',
                'task_name': task_name
            }
        else:
            connection['info'] = {
                'credentials': 'aggregator.json',
                'role': 'aggregator',
                'task_name': task_name
            }

    return connection


def get_aggregator_info(conn_type):

    if conn_type == 'flask':
        aggregator = {
            'ip': '192.168.0.231',
            'port': 5000
        }
    else:
        aggregator = {}

    return aggregator


def get_privacy():
    privacy = {
        'metrics': True
    }

    return privacy


# def get_mh():
#     metrics = {
#         'name': 'FileCheckpointHandler',
#         'path': 'ibmfl.aggregator.metric_service'
#     }
#     return metrics


def generate_ph_config(conn_type, is_party=False):
    if is_party:
        protocol_handler = {
            'name': 'PartyProtocolHandler',
            'path': 'ibmfl.party.party_protocol_handler'
        }
    else:
        protocol_handler = {
            'name': 'ProtoHandler',
            'path': 'ibmfl.aggregator.protohandler.proto_handler'
        }
    if conn_type == 'pubsub':
        protocol_handler['name'] += 'RabbitMQ'
    return protocol_handler


def generate_fusion_config(module):
    gen_fusion_config = getattr(module, 'get_fusion_config')
    return gen_fusion_config()


def generate_hp_config(model, module, num_parties):
    gen_hp_config = getattr(module, 'get_hyperparams')
    hp = gen_hp_config(model)
    hp['global']['num_parties'] = num_parties

    return hp


def generate_model_config(module, model, folder_configs, dataset, is_agg=False, party_id=0):
    get_model_config = getattr(module, 'get_model_config')
    model = get_model_config(folder_configs, dataset, is_agg, party_id, model=model)

    return model


def generate_lt_config(module, folder_configs=None):
    get_local_training_config = getattr(module, 'get_local_training_config')
    lt = get_local_training_config(configs_folder=folder_configs)

    return lt


def generate_datahandler_config(module, model, party_id, dataset, folder_data, is_agg=False):

    get_data_handler_config = getattr(module, 'get_data_handler_config')
    dh = get_data_handler_config(party_id, dataset, folder_data, is_agg, model=model)

    return dh


def generate_agg_config(module, model, num_parties, conn_type,  dataset,
                        folder_data, folder_configs, num_tiers, task_name=None, party_ip='127.0.0.1'):

    if not os.path.exists(folder_configs):
        os.makedirs(folder_configs)
    config_file = os.path.join(folder_configs, 'config_agg.yml')

    content = {
        'connection': generate_connection_config(conn_type, task_name=task_name, party_ip=party_ip),
        'model': generate_model_config(module, model, folder_configs, dataset),
        'fusion': generate_fusion_config(module),
        'hyperparams': generate_hp_config(model, module, num_parties),
        'protocol_handler': generate_ph_config(conn_type),
    }

    model_config = generate_model_config(module, model, folder_configs, dataset, True)
    data = generate_datahandler_config(module, model, 0, dataset, folder_data, True)

    model_config = generate_model_config(module, model, folder_configs, dataset, True)
    data = generate_datahandler_config(module, model, 0, dataset, folder_data, True)
    files_list = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    data_DA = []
    for tier in range(0,num_tiers):
        if dataset == 'emnist':
            data_folder = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_D{}.npz'.format(files_list[tier])
            data_DA.append(generate_datahandler_config(module, model, 0, dataset, data_folder, True))
        elif dataset == 'cifar10':
            data_folder = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/cifar10/balanced/cifar10_D{}.npz'.format(files_list[tier])
            data_DA.append(generate_datahandler_config(module, model, 0, dataset, data_folder, True))
    # data_folder_DA = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_DA.npz'
    # data_folder_DB = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_DB.npz'
    # data_folder_DC = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_DC.npz'
    # data_folder_DD = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_DD.npz'
    # data_folder_DA = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/cifar10/balanced/cifar10_DA.npz'
    # data_folder_DB = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/cifar10/balanced/cifar10_DB.npz'
    # data_DA = generate_datahandler_config(module, model, 0, dataset, data_folder_DA, True)
    # data_DB = generate_datahandler_config(module, model, 0, dataset, data_folder_DB, True)
    # data_DC = generate_datahandler_config(module, model, 0, dataset, data_folder_DC, True)
    # data_DD = generate_datahandler_config(module, model, 0, dataset, data_folder_DD, True)
    
    if model_config:
        content['model'] = model_config
    if data:
        content['data'] = data
        for tier in range(0,num_tiers):
            # print(files_list[tier])
            if dataset == 'emnist':
                content['data_D{}'.format(files_list[tier])] = data_DA[tier]
                content['data_D{}'.format(files_list[tier])]['info']['data_folder'] = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_D{}.npz'.format(files_list[tier])
                content['data_D{}'.format(files_list[tier])]['info']['npz_file'] = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/emnist/balanced/emnist_D{}.npz'.format(files_list[tier])
            elif dataset == 'cifar10':
                content['data_D{}'.format(files_list[tier])] = data_DA[tier]
                content['data_D{}'.format(files_list[tier])]['info']['data_folder'] = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/cifar10/balanced/cifar10_D{}.npz'.format(files_list[tier])
                content['data_D{}'.format(files_list[tier])]['info']['npz_file'] = '/home/ahmad/anaconda3/envs/PIFL/lib/python3.7/site-packages/ibmfl/incentive_IBMfl/examples/data/cifar10/balanced/cifar10_D{}.npz'.format(files_list[tier])
                 
        # content['data_DA'] = data_DA
        # content['data_DA']['info']['data_folder'] = data_folder_DA
        # content['data_DA']['info']['npz_file'] = data_folder_DA
        # content['data_DB'] = data_DB
        # content['data_DB']['info']['data_folder'] = data_folder_DB
        # content['data_DB']['info']['npz_file'] = data_folder_DB
        # content['data_DC'] = data_DC
        # content['data_DC']['info']['data_folder'] = data_folder_DC
        # content['data_DC']['info']['npz_file'] = data_folder_DC
        # content['data_DD'] = data_DD
        # content['data_DD']['info']['data_folder'] = data_folder_DD
        # content['data_DD']['info']['npz_file'] = data_folder_DD
        
    with open(config_file, 'w') as outfile:
        yaml.dump(content, outfile)

    print('Finished generating config file for aggregator. Files can be found in: ',
          os.path.abspath(os.path.join(folder_configs, 'config_agg.yml')))


def generate_party_config(module, model, num_parties, conn_type, dataset,
                          folder_data, folder_configs, num_tiers, task_name=None, party_ip='127.0.0.1'):

    for i in range(num_parties):
        config_file = os.path.join(
            folder_configs, 'config_party' + str(i) + '.yml')

        content = {
            'connection': generate_connection_config(conn_type, i, True, task_name=task_name, party_ip=party_ip),
            'data': generate_datahandler_config(module, model, i, dataset, folder_data),
            'model': generate_model_config(module, model, folder_configs, dataset, party_id=i),
            'protocol_handler': generate_ph_config(conn_type, True),
            'local_training': generate_lt_config(module, folder_configs=folder_configs),
            'aggregator': get_aggregator_info(conn_type),
            'privacy': get_privacy(),
            'hyperparams': {'tiers': num_tiers},
        }

        with open(config_file, 'w') as outfile:
            yaml.dump(content, outfile)

    print('Finished generating config file for parties. Files can be found in: ',
          os.path.abspath(os.path.join(folder_configs, 'config_party*.yml')))


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()
    #check_valid_folder_structure(parser)

    # Collect arguments
    num_parties = args.num_parties
    num_tiers = args.num_tiers
    dataset = args.dataset
    party_data_path = args.data_path
    config_path = args.config_path
    model = args.model
    fusion = args.fusion
    create_new = args.create_new
    exp_name = args.name
    conn_type = args.connection
    task_name = args.task_name
    context_path = args.context_path
    party_ip = args.party_ip

    # Create folder to save configs
    if config_path:
        if not os.path.exists(config_path):
            print('Config Path:{} does not exist.'.format(config_path))
            print('Creating {}'.format(config_path))
            try:
                os.makedirs(config_path, exist_ok=True)
            except OSError:
                print('Creating directory {} failed'.format(config_path))
                sys.exit(1)
        folder_configs = os.path.join(config_path, "configs")
    else:
        folder_configs = os.path.join("examples", "configs")
    if not model or model == "None" or model == "default":
        model = ''
        
    if create_new:
        folder_configs = os.path.join(
            folder_configs, exp_name if exp_name else str(int(time.time())))
    else:
        folder_configs = os.path.join(folder_configs, fusion, model)

    # Import and run generate_configs.py
    if context_path is not None and context_path != 'None':
        context = FL_CONTEXT.get(context_path) or None
        if context is not None:
            config_fusion = import_module('{}.{}.generate_configs'.format(context,fusion))
        else:
            print('Context path - {} is not correct, please check '.format(context_path))
            sys.exit(1)
    else:
        config_fusion = import_module('examples.{}.generate_configs'.format(fusion))

    generate_agg_config(config_fusion, model, num_parties, conn_type,
                        dataset, party_data_path, folder_configs, num_tiers, task_name, party_ip)
    generate_party_config(config_fusion, model, num_parties, conn_type,
                          dataset, party_data_path, folder_configs, num_tiers, task_name, party_ip)
