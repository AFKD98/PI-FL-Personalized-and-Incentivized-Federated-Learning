#!/usr/bin/env python3
import os
import sys
import csv
import time
import copy
import torch
import argparse
import numpy as np
import pandas as pd
import random
import pickle
from random import shuffle
from sys import getsizeof

fl_path = os.path.abspath('.')
if fl_path not in sys.path:
    sys.path.append(fl_path)

from ibmfl.util.datasets import load_nursery, load_mnist, load_emnist, load_adult, load_compas, load_german, \
    load_higgs, load_airline, load_diabetes, load_binovf, load_multovf, load_linovf, \
    load_simulated_federated_clustering, load_leaf_femnist, load_cifar10, load_wikipedia
from examples.constants import GENERATE_DATA_DESC, NUM_PARTIES_DESC, DATASET_DESC, PATH_DESC, PER_PARTY, \
    STRATIFY_DESC, FL_DATASETS, NEW_DESC, PER_PARTY_ERR, NAME_DESC, DATASET_DIST


def setup_parser():
    """
    Sets up the parser for Python script

    :return: a command line parser
    :rtype: argparse.ArgumentParser
    """
    p = argparse.ArgumentParser(description=GENERATE_DATA_DESC)
    p.add_argument("--num_parties", "-n", help=NUM_PARTIES_DESC,
                   type=int, required=True)
    p.add_argument("--dataset", "-d",
                   help=DATASET_DESC, required=True)
    p.add_argument("--data_path", "-p", help=PATH_DESC)
    p.add_argument("--points_per_party", "-pp", help=PER_PARTY,
                   nargs="+", type=int, required=True)
    p.add_argument("--stratify", "-s", help=STRATIFY_DESC, action="store_true")
    p.add_argument("--create_new", "-new", action="store_true", help=NEW_DESC)
    p.add_argument("--name", help=NAME_DESC)
    # To modify dataset distribution to non-iid, please use command like: -dd non-iid-d-2
    # d represents dirichlet
    # 2 represents the variable that controls the data heterogeneity, smaller
    # means higher data heterogeneity
    p.add_argument("--data_distribution", "-dd", help=DATASET_DIST)
    return p


def print_statistics(i, x_test_pi, x_train_pi, nb_labels, y_train_pi):
    print('Party_', i)
    print('nb_x_train: ', np.shape(x_train_pi),
          'nb_x_test: ', np.shape(x_test_pi))
    for l in range(nb_labels):
        print('* Label ', l, ' samples: ', (y_train_pi == l).sum())


def save_nursery_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Nursery party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    x_train = load_nursery(download_dir=dataset_folder)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_adult_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Adult party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    x_train = load_adult(download_dir=dataset_folder)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {group: counts[np.where(groups == group)[
            0][0]] / float(num_train) for group in groups}
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_german_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves German Credit Scorning party data
    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    x_train = load_german(download_dir=dataset_folder)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_compas_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Compas party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool``
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    x_train = load_compas(download_dir=dataset_folder)
    num_train = len(x_train.index)
    y_train = x_train['class'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)

def save_cifar10_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder, data_distribution):
    """
    Saves Cifar10 party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    :param data_distribution: data distribution type
    :type data_distribution: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_cifar10(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

    if data_distribution and 'non-iid' in data_distribution:
        data_distribution_list = data_distribution.split('-')
        data_distribution_mode = data_distribution_list[-2]
        beta = float(data_distribution_list[-1])

        if data_distribution_mode == 'd':
            data_split = dirchilet_non_iid(
                nb_dp_per_party=nb_dp_per_party,
                y_train=copy.deepcopy(y_train),
                unique_labels_num=num_labels,
                beta=beta
            )
        else:
            raise ValueError('wrong non-iid data distribution mode')

    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p = np.array(train_p)
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        # handle non-iid distribution for training data
        if data_distribution and 'non-iid' in data_distribution:
            train_indices = data_split[idx]
   
        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)

def dirchilet_non_iid(
    nb_dp_per_party,
    y_train,
    unique_labels_num,
    beta
):
    
    import torch

    if unique_labels_num < beta:
        raise ValueError('Please input correct dirichlet class number')
    
    num_party = len(nb_dp_per_party)
    num_train = np.shape(y_train)[0]
    # y_train = torch.from_numpy(y_train)
    data_split = {}
    dir = torch.distributions.dirichlet.Dirichlet(torch.tensor(beta).repeat(num_party))
    min_size = 0
    required_min_size = 10

    

    while min_size < required_min_size:
        data_split = [[] for _ in range(num_party)]
        for label_i in range(unique_labels_num):
            '''
            torch version
            '''
            # selected_train_idx = torch.where(y_train == label_i)[0]
            # proportions = dir.sample()
            # proportions = np.tensor(
            #     [p * (len(data_split_idx) < (num_train / num_party)) for p, data_split_idx in zip(proportions, data_split)])
            # proportions = proportions / proportions.sum()
            # # torch.cumsum累加最后一个纬度去分index
            # split_idx = (torch.cumsum(proportions, dim=-1) * len(selected_train_idx)).long().tolist()[:-1]
            # split_idx = torch.tensor_split(selected_train_idx, split_idx)
            # data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]

            '''
            np version
            '''
            selected_train_idx = np.where(y_train == label_i)[0]
            proportions = dir.sample().numpy()
            print(proportions)
            proportions = np.array(
                [p * (len(data_split_idx) < (num_train / num_party)) for p, data_split_idx in zip(proportions, data_split)])
            proportions = proportions / proportions.sum()
            # torch.cumsum累加最后一个纬度去分index
            split_idx = (np.cumsum(proportions, axis=-1) * len(selected_train_idx)).astype(np.int64).tolist()[:-1]
            split_idx = np.split(selected_train_idx, split_idx)
            data_split = [data_split_idx + idx.tolist() for data_split_idx, idx in zip(data_split, split_idx)]
        min_size = min([len(data_split_idx) for data_split_idx in data_split])
    data_split = {i: data_split[i] for i in range(num_party)}
    return data_split

def save_mnist_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder, data_distribution):
    """
    Saves MNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type data_path: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_mnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)

    if should_stratify:
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

    print(f'data_distribution: {data_distribution}')
    if data_distribution and 'non-iid' in data_distribution:
        data_distribution_list = data_distribution.split('-')
        data_distribution_mode = data_distribution_list[-2]
        beta = float(data_distribution_list[-1])

        if data_distribution_mode == 'd':
            data_split = dirchilet_non_iid(
                nb_dp_per_party=nb_dp_per_party,
                y_train=y_train,
                unique_labels_num=num_labels,
                beta=beta
            )
        else:
            raise ValueError('wrong non-iid data distribution mode')

    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(num_test / nb_parties), p=test_p)

        # handle non-iid distribution for training data
        if data_distribution and 'non-iid' in data_distribution:
            train_indices = data_split[idx]

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]
        print('shapes of all the arrays: ', x_train_pi.shape, y_train_pi.shape, x_test_pi.shape, y_test_pi.shape)

        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)

def save_emnist_party_data_with_distributions(nb_dp_per_party, should_stratify, party_folder, dataset_folder, data_distribution):
    """
    Saves EMNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type data_path: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    distributions = 4
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_emnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")
    repeats = 0
    classes = np.unique(y_train)
    #sort classes
    classes = np.sort(classes[:52])
    group_A = 50
    p_2 = 0.7
    p_1 = 0.3
    val_split = 0.2
    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)
    print('labels: %s', np.unique(y_test))
    print('classes: %s', classes)
    
    D_train_x = [[] for i in range(distributions)]
    D_train_y = [[] for i in range(distributions)]
    D_test_x = [[] for i in range(distributions)]
    D_test_y = [[] for i in range(distributions)]
    print(D_train_x)
    #Split data into two datasets
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    #Remove digits from train
    x_train_letters = []
    y_train_letters = []
    class_cutoff = len(classes)/distributions
    current_distrbution = 0
    for x,y in zip(x_train, y_train):
        if int(y) > 9:
            x_train_letters.append(x)
            y_train_letters.append(y-10)
            #Split data into two datasets
            D_train_x[int(int(y-10)/class_cutoff)-1].append(x)
            D_train_y[int(int(y-10)/class_cutoff)-1].append(y-10)

    

    
    #Remove digits from test
    x_test_letters = []
    y_test_letters = []
    current_distrbution = 0
    
    for x,y in zip(x_test, y_test):
        if int(y) > 9:
            x_test_letters.append(x)
            y_test_letters.append(y-10)
            #Split data into two datasets
            D_test_x[int(int(y-10)/class_cutoff)-1].append(x)
            D_test_y[int(int(y-10)/class_cutoff)-1].append(y-10)
    
    #Convert to numpy arrays
    x_train = np.asarray(x_train_letters)
    y_train = np.asarray(y_train_letters)
    x_test = np.asarray(x_test_letters)
    y_test = np.asarray(y_test_letters)
    for i in range(distributions):
        D_train_x[i] = np.asarray(D_train_x[i])
        D_train_y[i] = np.asarray(D_train_y[i])
        D_test_x[i] = np.asarray(D_test_x[i])
        D_test_y[i] = np.asarray(D_test_y[i])
    
    train_class_count = np.bincount(y_train)
    test_class_count = np.bincount(y_test)
    # Key - Class Number (0-9), Value: index list of 
    # all images belonging to that class
    train_indices_by_class, train_common_indices_by_class = {}, {}
    test_indices_by_class = {}
    # Key - client number, Value - List of image indices for that client
    train_idx_per_client = {}
    val_idx_per_client = {}
    test_idx_per_client = {}

    # Train dataset
    for img_index, class_num in enumerate(y_train):
        train_indices_by_class.setdefault(class_num, []).append(img_index)
        train_common_indices_by_class.setdefault(class_num, []).append(img_index)
        
    for img_index, class_num in enumerate(y_test):
        test_indices_by_class.setdefault(class_num, []).append(img_index)
    
    # Assign classes to clients
    current_distrbution = 0
    classes_for_client = []
    end = int(len(classes)/distributions)
    start = 0
    
    client_divisions = nb_parties/distributions
    for client_num, dp in enumerate(nb_dp_per_party):
        
        if client_divisions > 1 and client_num > 1:
            if client_num % client_divisions == 0:
                start = end
                end = int(end + len(classes)/distributions)
        print("start: ", start)
        print("end: ", end)
        classes_for_client.append(list(range(start, end)))

    print("classes_for_client: ", classes_for_client)
            
    num_clients = len(nb_dp_per_party)
    
    for client_num, dp in enumerate(nb_dp_per_party):
        p_2= (0.005 + (client_num)* 0.01)
        p_1= (0.995 - (client_num)* 0.01)
        # p_2= random.uniform(0, 1)
        # p_1 = 1 - p_2
        print("p_2: ", p_2)
        print("p_1: ", p_1)
        
        classes_assigned = classes_for_client[client_num]
        train_idx_per_client[client_num] = []
        val_idx_per_client[client_num] = []
        test_idx_per_client[client_num] = []
                    
        for class_no in classes:
            num_train_imgs_perclass = train_class_count[class_no]
            num_test_imgs_perclass = test_class_count[class_no]
            num_samples_max = 1000
            if class_no in classes_assigned:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_2/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_1/num_clients), num_samples_max)
                # print('train_indices_by_class[class_no]: ', train_indices_by_class[class_no])
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices
                
            else:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_1/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_2/num_clients), num_samples_max)
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices

            remaining = list(set(train_common_indices_by_class[class_no]) - set(train_indices))
            num_repeat_choices = min(len(remaining), num_train_imgs_perclass * repeats)
            train_indices = np.append(train_indices, np.random.choice(
                remaining, num_repeat_choices, replace=False)).astype(int)
            train_idx_per_client[client_num].extend(train_indices)
            # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            test_idx_per_client[client_num].extend(test_indices)
            
        val_idx_per_client[client_num] = np.random.choice(train_idx_per_client[client_num], int(len(train_idx_per_client[client_num])*val_split), replace=False)
        train_idx_per_client[client_num] = list(set(train_idx_per_client[client_num]) - set(val_idx_per_client[client_num]))
        print("Client: ", client_num, " Class: ", class_no, " Train: ", len(train_indices), " Test: ", len(test_indices), 'test class: ', class_no)
        # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            
        # for idx, dp in enumerate(nb_dp_per_party):
        x_train_pi = x_train[train_idx_per_client[client_num]].reshape(len(train_idx_per_client[client_num]), 28, 28)
        y_train_pi = y_train[train_idx_per_client[client_num]]
        x_val_pi = x_train[val_idx_per_client[client_num]].reshape(len(val_idx_per_client[client_num]), 28, 28)
        y_val_pi = y_train[val_idx_per_client[client_num]]
        x_test_pi = x_test[test_idx_per_client[client_num]].reshape(len(test_idx_per_client[client_num]), 28, 28)
        y_test_pi = y_test[test_idx_per_client[client_num]]
        print_statistics(client_num, x_test_pi, x_train_pi, num_labels, y_train_pi)
        print(len(train_idx_per_client[client_num]), len(val_idx_per_client[client_num]), len(test_idx_per_client[client_num]))
        # Now put it all in an npz
        print(type(x_train_pi[0]), type(y_train_pi), type(x_test_pi), type(y_test_pi))
        name_file = 'data_party' + str(client_num) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        print('shapes of all the arrays: ', x_train_pi.shape, y_train_pi.shape, x_test_pi.shape, y_test_pi.shape)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_val=x_val_pi, y_val=y_val_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print('Finished! :) Data saved in ', party_folder)
    
    if not os.path.exists('examples/data/emnist/balanced'):
        os.makedirs('examples/data/emnist/balanced')
    
    file_names = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    for distribution in range(0, distributions):
        
        # idx = np.random.choice(np.arange(len(D_test_x)), 2000, replace=False)
        # D_train_x[distribution] = D_train_x[distribution][idx]
        # D_train_y[distribution] = D_train_y[distribution][idx]
        # D_test_x[distribution] = D_test_x[distribution][idx]
        # D_test_y[distribution] = D_test_y[distribution][idx]
        
        print('D_train_x[distribution].shape: ', D_train_x[distribution].shape)
        np.savez('examples/data/emnist/balanced/emnist_D{}.npz'.format(file_names[distribution]), 
        x_train=D_train_x[distribution], y_train=D_train_y[distribution], 
        x_test=D_test_x[distribution], y_test=D_test_y[distribution])
    
    idx = np.random.choice(np.arange(len(y_train)), 2000, replace=False)
    x_train = x_train[idx]
    y_train = y_train[idx]
    idx = np.random.choice(np.arange(len(y_test)), 2000, replace=False)
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    np.savez('examples/datasets/emnist.npz', 
        x_train=x_train, y_train=y_train, 
        x_test=x_test, y_test=y_test)
    
def save_emnist_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder, data_distribution):
    """
    Saves EMNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type data_path: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    distributions = 2
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_emnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")
    repeats = 0
    classes = np.unique(y_train)
    #sort classes
    classes = np.sort(classes[:52])
    group_A = 50
    p_2 = 0.7
    p_1 = 0.3
    val_split = 0.2
    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)
    print('labels: %s', np.unique(y_test))
    print('classes: %s', classes)
    
    #Split data into two datasets
    DA_train_x = []
    DA_train_y = []
    DA_test_x = []
    DA_test_y = []
    DB_train_x = []
    DB_train_y = []
    DB_test_x = []
    DB_test_y = []
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    #Remove digits from train
    x_train_letters = []
    y_train_letters = []
    for x,y in zip(x_train, y_train):
        if int(y) > 9:
            x_train_letters.append(x)
            y_train_letters.append(y-10)
            #Split data into two datasets
            if int(y) < len(classes)/2:
                DA_train_x.append(x)
                DA_train_y.append(y-10)
            else:
                DB_train_x.append(x)
                DB_train_y.append(y-10)

    

    
    #Remove digits from test
    x_test_letters = []
    y_test_letters = []
    for x,y in zip(x_test, y_test):
        if int(y) > 9:
            x_test_letters.append(x)
            y_test_letters.append(y-10)
            if int(y) < len(classes)/2:
                DA_test_x.append(x)
                DA_test_y.append(y-10)
            else:
                DB_test_x.append(x)
                DB_test_y.append(y-10)
    
    #Convert to numpy arrays
    x_train = np.asarray(x_train_letters)
    y_train = np.asarray(y_train_letters)
    x_test = np.asarray(x_test_letters)
    y_test = np.asarray(y_test_letters)
    DA_train_x = np.asarray(DA_train_x)
    DA_train_y = np.asarray(DA_train_y)
    DA_test_x = np.asarray(DA_test_x)
    DA_test_y = np.asarray(DA_test_y)
    DB_train_x = np.asarray(DB_train_x)
    DB_train_y = np.asarray(DB_train_y)
    DB_test_x = np.asarray(DB_test_x)
    DB_test_y = np.asarray(DB_test_y)
    
    train_class_count = np.bincount(y_train)
    test_class_count = np.bincount(y_test)
    # Key - Class Number (0-9), Value: index list of 
    # all images belonging to that class
    train_indices_by_class, train_common_indices_by_class = {}, {}
    test_indices_by_class = {}
    # Key - client number, Value - List of image indices for that client
    train_idx_per_client = {}
    val_idx_per_client = {}
    test_idx_per_client = {}

    # Train dataset
    for img_index, class_num in enumerate(y_train):
        train_indices_by_class.setdefault(class_num, []).append(img_index)
        train_common_indices_by_class.setdefault(class_num, []).append(img_index)
        
    for img_index, class_num in enumerate(y_test):
        test_indices_by_class.setdefault(class_num, []).append(img_index)
    
    # Assign classes to clients
    classes_for_client = []
    for client_num, dp in enumerate(nb_dp_per_party):
        if client_num < group_A:
            classes_for_client.append(list(range(0, 26)))
        else:
            classes_for_client.append(list(range(26, 52)))
    
    num_clients = len(nb_dp_per_party)
    
    for client_num, dp in enumerate(nb_dp_per_party):
        p_2= (0.005 + (client_num)* 0.01)
        # p_1= (0.995 - (client_num)* 0.01)
        # p_2= random.uniform(0, 1)
        p_1 = 1 - p_2
        print("p_2: ", p_2)
        print("p_1: ", p_1)
        
        classes_assigned = classes_for_client[client_num]
        train_idx_per_client[client_num] = []
        val_idx_per_client[client_num] = []
        test_idx_per_client[client_num] = []
                    
        for class_no in classes:
            num_train_imgs_perclass = train_class_count[class_no]
            num_test_imgs_perclass = test_class_count[class_no]
            num_samples_max = 1000
            if class_no in classes_assigned:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_2/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_1/num_clients), num_samples_max)
                # print('train_indices_by_class[class_no]: ', train_indices_by_class[class_no])
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices
                
            else:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_1/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_2/num_clients), num_samples_max)
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices

            remaining = list(set(train_common_indices_by_class[class_no]) - set(train_indices))
            num_repeat_choices = min(len(remaining), num_train_imgs_perclass * repeats)
            train_indices = np.append(train_indices, np.random.choice(
                remaining, num_repeat_choices, replace=False)).astype(int)
            train_idx_per_client[client_num].extend(train_indices)
            # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            test_idx_per_client[client_num].extend(test_indices)
            
        val_idx_per_client[client_num] = np.random.choice(train_idx_per_client[client_num], int(len(train_idx_per_client[client_num])*val_split), replace=False)
        train_idx_per_client[client_num] = list(set(train_idx_per_client[client_num]) - set(val_idx_per_client[client_num]))
        print("Client: ", client_num, " Class: ", class_no, " Train: ", len(train_indices), " Test: ", len(test_indices), 'test class: ', class_no)
        # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            
        # for idx, dp in enumerate(nb_dp_per_party):
        x_train_pi = x_train[train_idx_per_client[client_num]].reshape(len(train_idx_per_client[client_num]), 28, 28)
        y_train_pi = y_train[train_idx_per_client[client_num]]
        x_val_pi = x_train[val_idx_per_client[client_num]].reshape(len(val_idx_per_client[client_num]), 28, 28)
        y_val_pi = y_train[val_idx_per_client[client_num]]
        x_test_pi = x_test[test_idx_per_client[client_num]].reshape(len(test_idx_per_client[client_num]), 28, 28)
        y_test_pi = y_test[test_idx_per_client[client_num]]
        print_statistics(client_num, x_test_pi, x_train_pi, num_labels, y_train_pi)
        print(len(train_idx_per_client[client_num]), len(val_idx_per_client[client_num]), len(test_idx_per_client[client_num]))
        # Now put it all in an npz
        print(type(x_train_pi[0]), type(y_train_pi), type(x_test_pi), type(y_test_pi))
        name_file = 'data_party' + str(client_num) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        print('shapes of all the arrays: ', x_train_pi.shape, y_train_pi.shape, x_test_pi.shape, y_test_pi.shape)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_val=x_val_pi, y_val=y_val_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print('Finished! :) Data saved in ', party_folder)
        
    idx = np.random.choice(np.arange(len(DA_test_x)), 2000, replace=False)
    DA_train_x = DA_train_x[idx]
    DA_train_y = DA_train_y[idx]
    DA_test_x = DA_test_x[idx]
    DA_test_y = DA_test_y[idx]
    idx = np.random.choice(np.arange(len(DB_test_x)), 2000, replace=False)
    DB_train_x = DB_train_x[idx]
    DB_train_y = DB_train_y[idx]
    DB_test_x = DB_test_x[idx]
    DB_test_y = DB_test_y[idx]
    if not os.path.exists('examples/data/emnist/balanced'):
        os.makedirs('examples/data/emnist/balanced')
    
    np.savez('examples/data/emnist/balanced/emnist_DA.npz', 
    x_train=DA_train_x, y_train=DA_train_y, 
    x_test=DA_test_x, y_test=DA_test_y)
    
    np.savez('examples/data/emnist/balanced/emnist_DB.npz', 
        x_train=DB_train_x, y_train=DB_train_y, 
        x_test=DB_test_x, y_test=DB_test_y)
    
    idx = np.random.choice(np.arange(len(y_train)), 2000, replace=False)
    x_train = x_train[idx]
    y_train = y_train[idx]
    idx = np.random.choice(np.arange(len(y_test)), 2000, replace=False)
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    np.savez('examples/datasets/emnist.npz', 
        x_train=x_train, y_train=y_train, 
        x_test=x_test, y_test=y_test)
    
    
def save_emnist_party_data_non_IID(nb_dp_per_party, should_stratify, party_folder, dataset_folder, data_distribution):
    """
    Saves EMNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type data_path: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    distributions = 2
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    (x_train, y_train), (x_test, y_test) = load_emnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")
    repeats = 0
    classes = np.unique(y_train)
    #sort classes
    classes = np.sort(classes[:52])
    group_A = 50
    p_2 = 0.7
    p_1 = 0.3
    val_split = 0.2
    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]
    nb_parties = len(nb_dp_per_party)
    print('labels: %s', np.unique(y_test))
    print('classes: %s', classes)
    
    #Split data into two datasets
    DA_train_x = []
    DA_train_y = []
    DA_test_x = []
    DA_test_y = []
    DB_train_x = []
    DB_train_y = []
    DB_test_x = []
    DB_test_y = []
    x_train = x_train.numpy()
    y_train = y_train.numpy()
    x_test = x_test.numpy()
    y_test = y_test.numpy()
    #Remove digits from train
    x_train_letters = []
    y_train_letters = []
    for x,y in zip(x_train, y_train):
        if int(y) > 9:
            x_train_letters.append(x)
            y_train_letters.append(y-10)
            #Split data into two datasets
            if int(y) < len(classes)/2:
                DA_train_x.append(x)
                DA_train_y.append(y-10)
            else:
                DB_train_x.append(x)
                DB_train_y.append(y-10)

    

    
    #Remove digits from test
    x_test_letters = []
    y_test_letters = []
    for x,y in zip(x_test, y_test):
        if int(y) > 9:
            x_test_letters.append(x)
            y_test_letters.append(y-10)
            if int(y) < len(classes)/2:
                DA_test_x.append(x)
                DA_test_y.append(y-10)
            else:
                DB_test_x.append(x)
                DB_test_y.append(y-10)
    
    #Convert to numpy arrays
    x_train = np.asarray(x_train_letters)
    y_train = np.asarray(y_train_letters)
    x_test = np.asarray(x_test_letters)
    y_test = np.asarray(y_test_letters)
    DA_train_x = np.asarray(DA_train_x)
    DA_train_y = np.asarray(DA_train_y)
    DA_test_x = np.asarray(DA_test_x)
    DA_test_y = np.asarray(DA_test_y)
    DB_train_x = np.asarray(DB_train_x)
    DB_train_y = np.asarray(DB_train_y)
    DB_test_x = np.asarray(DB_test_x)
    DB_test_y = np.asarray(DB_test_y)
    
    train_class_count = np.bincount(y_train)
    test_class_count = np.bincount(y_test)
    # Key - Class Number (0-9), Value: index list of 
    # all images belonging to that class
    train_indices_by_class, train_common_indices_by_class = {}, {}
    test_indices_by_class = {}
    # Key - client number, Value - List of image indices for that client
    train_idx_per_client = {}
    val_idx_per_client = {}
    test_idx_per_client = {}

    # Train dataset
    for img_index, class_num in enumerate(y_train):
        train_indices_by_class.setdefault(class_num, []).append(img_index)
        train_common_indices_by_class.setdefault(class_num, []).append(img_index)
        
    for img_index, class_num in enumerate(y_test):
        test_indices_by_class.setdefault(class_num, []).append(img_index)
    
    # Assign classes to clients
    classes_for_client = []
    for client_num, dp in enumerate(nb_dp_per_party):
        if client_num < group_A:
            classes_for_client.append(list(range(0, 26)))
        else:
            classes_for_client.append(list(range(26, 52)))
    
    num_clients = len(nb_dp_per_party)
    
    for client_num, dp in enumerate(nb_dp_per_party):
        p_2= (0.005 + (client_num)* 0.01)
        # p_1= (0.995 - (client_num)* 0.01)
        # p_2= random.uniform(0, 1)
        p_1 = 1 - p_2
        print("p_2: ", p_2)
        print("p_1: ", p_1)
        
        classes_assigned = classes_for_client[client_num]
        train_idx_per_client[client_num] = []
        val_idx_per_client[client_num] = []
        test_idx_per_client[client_num] = []
                    
        for class_no in classes:
            num_train_imgs_perclass = train_class_count[class_no]
            num_test_imgs_perclass = test_class_count[class_no]
            num_samples_max = 1000
            if class_no in classes_assigned:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_2/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_1/num_clients), num_samples_max)
                # print('train_indices_by_class[class_no]: ', train_indices_by_class[class_no])
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices
                
            else:
                num_train_imgs_perclass = min(int(train_class_count[class_no]*p_1/num_clients), num_samples_max)
                num_test_imgs_perclass = min(int(test_class_count[class_no]*p_2/num_clients), num_samples_max)
                train_indices = np.random.choice(train_indices_by_class[class_no], num_train_imgs_perclass, replace=False)
                leftover_train_indices = list(set(train_indices_by_class[class_no]) - set(train_indices))
                train_indices_by_class[class_no] = leftover_train_indices
                test_indices = np.random.choice(test_indices_by_class[class_no], num_test_imgs_perclass, replace=False)
                leftover_test_indices = list(set(test_indices_by_class[class_no]) - set(test_indices))
                test_indices_by_class[class_no] = leftover_test_indices

            remaining = list(set(train_common_indices_by_class[class_no]) - set(train_indices))
            num_repeat_choices = min(len(remaining), num_train_imgs_perclass * repeats)
            train_indices = np.append(train_indices, np.random.choice(
                remaining, num_repeat_choices, replace=False)).astype(int)
            train_idx_per_client[client_num].extend(train_indices)
            # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            test_idx_per_client[client_num].extend(test_indices)
            
        val_idx_per_client[client_num] = np.random.choice(train_idx_per_client[client_num], int(len(train_idx_per_client[client_num])*val_split), replace=False)
        train_idx_per_client[client_num] = list(set(train_idx_per_client[client_num]) - set(val_idx_per_client[client_num]))
        print("Client: ", client_num, " Class: ", class_no, " Train: ", len(train_indices), " Test: ", len(test_indices), 'test class: ', class_no)
        # print('len(train_idx_per_client[client_num]): ', len(train_idx_per_client[client_num]))
            
        # for idx, dp in enumerate(nb_dp_per_party):
        x_train_pi = x_train[train_idx_per_client[client_num]].reshape(len(train_idx_per_client[client_num]), 28, 28)
        y_train_pi = y_train[train_idx_per_client[client_num]]
        x_val_pi = x_train[val_idx_per_client[client_num]].reshape(len(val_idx_per_client[client_num]), 28, 28)
        y_val_pi = y_train[val_idx_per_client[client_num]]
        x_test_pi = x_test[test_idx_per_client[client_num]].reshape(len(test_idx_per_client[client_num]), 28, 28)
        y_test_pi = y_test[test_idx_per_client[client_num]]
        print_statistics(client_num, x_test_pi, x_train_pi, num_labels, y_train_pi)
        print(len(train_idx_per_client[client_num]), len(val_idx_per_client[client_num]), len(test_idx_per_client[client_num]))
        # Now put it all in an npz
        print(type(x_train_pi[0]), type(y_train_pi), type(x_test_pi), type(y_test_pi))
        name_file = 'data_party' + str(client_num) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        print('shapes of all the arrays: ', x_train_pi.shape, y_train_pi.shape, x_test_pi.shape, y_test_pi.shape)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_val=x_val_pi, y_val=y_val_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print('Finished! :) Data saved in ', party_folder)
        
    idx = np.random.choice(np.arange(len(DA_test_x)), 2000, replace=False)
    DA_train_x = DA_train_x[idx]
    DA_train_y = DA_train_y[idx]
    DA_test_x = DA_test_x[idx]
    DA_test_y = DA_test_y[idx]
    idx = np.random.choice(np.arange(len(DB_test_x)), 2000, replace=False)
    DB_train_x = DB_train_x[idx]
    DB_train_y = DB_train_y[idx]
    DB_test_x = DB_test_x[idx]
    DB_test_y = DB_test_y[idx]
    if not os.path.exists('examples/data/emnist/balanced'):
        os.makedirs('examples/data/emnist/balanced')
    
    np.savez('examples/data/emnist/balanced/emnist_DA.npz', 
    x_train=DA_train_x, y_train=DA_train_y, 
    x_test=DA_test_x, y_test=DA_test_y)
    
    np.savez('examples/data/emnist/balanced/emnist_DB.npz', 
        x_train=DB_train_x, y_train=DB_train_y, 
        x_test=DB_test_x, y_test=DB_test_y)
    
    idx = np.random.choice(np.arange(len(y_train)), 2000, replace=False)
    x_train = x_train[idx]
    y_train = y_train[idx]
    idx = np.random.choice(np.arange(len(y_test)), 2000, replace=False)
    x_test = x_test[idx]
    y_test = y_test[idx]
    
    np.savez('examples/datasets/emnist.npz', 
        x_train=x_train, y_train=y_train, 
        x_test=x_test, y_test=y_test)
     
    
    

def save_higgs_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Higgs Boson party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    x, y = load_higgs(dataset_folder)
    num_train = len(x)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_airline_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Airline Delay party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    X, y = load_airline(dataset_folder)
    num_train = len(X)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    for i, dp in enumerate(nb_dp_per_party):
        '''
        # Even Parties Have Biased Dataset, Odd Are Randomly Sampled
        if not should_stratify and i % 2 == 0:
            # Unbalanced Dataset
            niid_prob = {label: 1.0 / num_train * 2 for label in labels}

            p_list = np.array([niid_prob[y[idx]] for idx in range(num_train)])
            p_list /= np.sum(p_list)
        else:
            # Regular Dataset
            p_list = np.array([probs[y[idx]] for idx in range(num_train)])
            p_list /= np.sum(p_list)
        '''
        
        # Regular Dataset
        p_list = np.array([probs[y[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_diabetes_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Diabetes party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    x_train = load_diabetes(dataset_folder)
    num_train = len(x_train)
    y_train = x_train['readmitted'].values.tolist()
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {
            group: counts[np.where(groups == group)[0][0]] / float(num_train) for group in groups
        }
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()
        # Use indices for data/classification subset
        x_train_pi = x_train.iloc[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        with open(name_file, 'w') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(x_train_pi)

        x_train_pi.to_csv(path_or_buf=name_file, index=None)

    print('Finished! :) Data saved in', party_folder)


def save_binovf_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Binary Overfit party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    X, y = load_binovf()
    num_train = len(X)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    p_list = np.array([probs[y[idx]] for idx in range(num_train)])
    p_list /= np.sum(p_list)
    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_multovf_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves Multiclass Overfit party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    x_train, y_train = load_multovf()
    num_train = len(x_train)
    labels, counts = np.unique(y_train, return_counts=True)

    if should_stratify:
        strat_col = y_train
        groups, counts = np.unique(strat_col, return_counts=True)
        # to use custom proportions, replace probs with a dictionary where key:value pairs are label:proportion
        probs = {group: counts[np.where(groups == group)[
            0][0]] / float(num_train) for group in groups}
        p_list = np.array([probs[strat_col[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    else:
        probs = {label: 1.0 / num_train for label in labels}
        p_list = np.array([probs[y_train[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in x_train[indices, :].astype(str)]
        y_part = y_train[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_linovf_party_data(nb_dp_per_party, party_folder, dataset_folder):
    """
    Saves Linear Overfit party data (For Regression)
    Data stratification is not supported in this function.

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    x_train, y_train = load_linovf()
    num_train = len(x_train)

    for i, dp in enumerate(nb_dp_per_party):
        # Create variable for indices
        indices = np.random.choice(num_train, dp)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [item for item in x_train[indices].astype(str)]
        y_part = y_train[indices]

        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(y_part[i])+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)

def save_femnist_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder):
    """
    Saves LEAF-FEMNIST party data

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`, if any value in list is -1, use femnist's default distribution
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :return: None
    :rtype: None
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    dataset_folder = os.path.join(dataset_folder, "femnist")

    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    num_parties = len(nb_dp_per_party)
    # FEMNIST's default data distribution based on LEAF
    if -1 in nb_dp_per_party :
        print("Generating dataset based on FEMNIST's default data distribution...")
        partywise_data = load_leaf_femnist(download_dir=dataset_folder, orig_dist=True)
        for idx, (_, data) in enumerate(partywise_data.items()):
            if idx >= num_parties:
                break
            train_indices = np.random.choice(len(data['x']), int(len(data['x']) * 0.9), replace=False)
            test_indices = [i for i in range(len(data['x'])) if i not in train_indices]
            x_train_pi = np.array([data['x'][i] for i in train_indices])
            y_train_pi = np.array([data['y'][i] for i in train_indices])
            x_test_pi = np.array([data['x'][i] for i in test_indices])
            y_test_pi = np.array([data['y'][i] for i in test_indices])

            # Now put it all in an npz
            name_file = 'data_party' + str(idx) + '.npz'
            name_file = os.path.join(party_folder, name_file)
            np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                    x_test=x_test_pi, y_test=y_test_pi)
            print_statistics(idx, x_test_pi, x_train_pi, 62, y_train_pi)
            print('Finished! :) Data saved in ', party_folder)
        return

    (x_train, y_train), (x_test, y_test) = load_leaf_femnist(download_dir=dataset_folder)
    labels, train_counts = np.unique(y_train, return_counts=True)
    te_labels, test_counts = np.unique(y_test, return_counts=True)
    if np.all(np.isin(labels, te_labels)):
        print("Warning: test set and train set contain different labels")

    num_train = np.shape(y_train)[0]
    num_test = np.shape(y_test)[0]
    num_labels = np.shape(np.unique(y_test))[0]

    # Synthetically distributed FEMNIST
    if should_stratify:
        print("Generating non-iid FEMNIST distribution...")
        # Sample according to source label distribution
        train_probs = {
            label: train_counts[label] / float(num_train) for label in labels}
        test_probs = {label: test_counts[label] /
                      float(num_test) for label in te_labels}
    else:
        print("Generating iid FEMNIST distribution...")
        # Sample uniformly
        train_probs = {label: 1.0 / len(labels) for label in labels}
        test_probs = {label: 1.0 / len(te_labels) for label in te_labels}

    for idx, dp in enumerate(nb_dp_per_party):
        train_p = np.array([train_probs[y_train[idx]]
                            for idx in range(num_train)])
        train_p /= np.sum(train_p)
        train_indices = np.random.choice(num_train, dp, p=train_p)
        test_p = np.array([test_probs[y_test[idx]] for idx in range(num_test)])
        test_p /= np.sum(test_p)

        # Split test evenly
        test_indices = np.random.choice(
            num_test, int(dp * 0.1), p=test_p)

        x_train_pi = x_train[train_indices]
        y_train_pi = y_train[train_indices]
        x_test_pi = x_test[test_indices]
        y_test_pi = y_test[test_indices]
        # Now put it all in an npz
        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_pi, y_train=y_train_pi,
                 x_test=x_test_pi, y_test=y_test_pi)

        print_statistics(idx, x_test_pi, x_train_pi, num_labels, y_train_pi)

        print('Finished! :) Data saved in ', party_folder)

def save_federated_clustering_data(nb_dp_per_party, party_folder):
    """
    Saves simulated federated clustering dataset for unsupervised federated
    learning setting

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    """

    num_clients = len(nb_dp_per_party)
    # Same number of data points are generated for each party
    nb_datapoints = nb_dp_per_party[0]
    # use true clusters depending on the number of clients
    # this prevents creating too many global centroids but fewer
    # cumulative local centroids
    true_clusters = 5 * num_clients

    kwargs = {
        "J": num_clients,
        "M": nb_datapoints,
        "L": true_clusters
    }

    # data returned is (J, M, D=100) dimensions
    data = load_simulated_federated_clustering(**kwargs)

    for idx in range(num_clients):
        x_train_np = np.array(data[idx])
        x_test_np = x_train_np      # Duplicating x_train to x_test

        name_file = 'data_party' + str(idx) + '.npz'
        name_file = os.path.join(party_folder, name_file)
        np.savez(name_file, x_train=x_train_np, x_test=x_test_np)

        print('Finished! :) Data saved in ', party_folder)

def save_party_data(nb_dp_per_party, should_stratify, party_folder, dataset_folder, dataset):
    """
    Loads a generate dataset saved as in csv format and creates parties local datasets
    as specified.

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param should_stratify: True if data should be assigned proportional to source class distributions
    :type should_stratify: `bool`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    :param dataset: the name of the csv file
    :type dataset: `str`
    """
    dataset_folder = os.path.join(dataset_folder, dataset) + '.csv'
    print("Loading the original dataset from: " + dataset_folder)

    try:
        # if no header
        data = pd.read_csv(dataset_folder, header=None).to_numpy()
        X, y = data[:, :-1], data[:, -1].astype('int')
    except Exception as ex:
        print(ex)
        print("Warning: please ensure the provided dataset is in .csv format.")
        print("Please ensure that the class labels are provided in the last column.")
        print("Warning: please ensure that the class labels are provided as numbers.")
        print("Loading the dataset assuming the header is provided in the 1st column.")
        data = pd.read_csv(dataset_folder, header=1).to_numpy()
        X, y = data[:, :-1], data[:, -1].astype('int')
        
    num_train = len(X)
    labels, counts = np.unique(y, return_counts=True)

    if should_stratify:
        probs = {label: counts[np.where(labels == label)[
            0][0]] / float(num_train) for label in labels}
    else:
        probs = {label: 1.0 / num_train for label in labels}

    for i, dp in enumerate(nb_dp_per_party):

        # Regular Dataset
        p_list = np.array([probs[y[idx]] for idx in range(num_train)])
        p_list /= np.sum(p_list)

        indices = np.random.choice(num_train, dp, p=p_list)
        indices = indices.tolist()

        # Use indices for data/classification subset
        x_part = [','.join(item) for item in X[indices, :].astype(str)]
        y_part = y[indices]

        # Write to File
        name_file = 'data_party' + str(i) + '.csv'
        name_file = os.path.join(party_folder, name_file)
        out = open(name_file, 'w')
        for i in range(len(x_part)):
            out.write(x_part[i]+','+str(int(y_part[i]))+'\n')
        out.close()

    print('Finished! :) Data saved in', party_folder)


def save_wikipedia_party_data(nb_dp_per_party, party_folder, dataset_folder):
    """
    Saves Wikipedia party data for Doc2Vec

    :param nb_dp_per_party: the number of data points each party should have
    :type nb_dp_per_party: `list[int]`
    :param party_folder: folder to save party data
    :type party_folder: `str`
    :param dataset_folder: folder to save dataset
    :type dataset_folder: `str`
    """
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    total_samples = 0
    for num_samples in nb_dp_per_party:
        total_samples += num_samples

    x = load_wikipedia(total_samples)
    shuffle(x)

    start_index = 0
    for i, dp in enumerate(nb_dp_per_party):
        end_index = start_index + dp
        party_sample = x[start_index: end_index]

        name_file = 'data_party' + str(i) + '.pickle'
        name_file = os.path.join(party_folder, name_file)

        with open(name_file, 'wb') as file:
            pickle.dump(party_sample, file)

        start_index = end_index

    print('Finished! :) Data saved in', party_folder)


if __name__ == '__main__':
    # Parse command line options
    parser = setup_parser()
    args = parser.parse_args()

    # Collect arguments
    num_parties = args.num_parties
    dataset = args.dataset
    data_path = args.data_path
    points_per_party = args.points_per_party
    stratify = args.stratify
    create_new = args.create_new
    exp_name = args.name
    data_distribution = args.data_distribution
    print(f'generate data argument: {args}')
    # Check for errors
    if len(points_per_party) == 1:
        points_per_party = [points_per_party[0] for _ in range(num_parties)]
    elif len(points_per_party) != num_parties:
        parser.error(PER_PARTY_ERR)

    if data_path is not None:
        if not os.path.exists(data_path):
                print('Data Path:{} does not exist.'.format(data_path))
                print('Creating {}'.format(data_path))
                try:
                    os.makedirs(data_path, exist_ok=True)
                except OSError:
                    print('Creating directory {} failed'.format(data_path))
                    sys.exit(1)
        folder_party_data = os.path.join(data_path, "data")
        folder_dataset = os.path.join(data_path, "datasets")
    else:
        folder_party_data = os.path.join("examples", "data")
        folder_dataset = os.path.join("examples", "datasets")

    strat = 'balanced' if stratify else 'random'
    if args.dataset == 'femnist' and -1 in points_per_party:
        strat = 'orig_dist'

    if create_new:
        folder_party_data = os.path.join(folder_party_data, exp_name if exp_name else str(
            int(time.time())) + '_' + strat)
    else:
        folder_party_data = os.path.join(folder_party_data, dataset, strat)

    if not os.path.exists(folder_party_data):
        os.makedirs(folder_party_data)
    else:
        # clear folder of old data
        for f_name in os.listdir(folder_party_data):
            f_path = os.path.join(folder_party_data, f_name)
            if os.path.isfile(f_path):
                os.unlink(f_path)

    # Save new files
    if dataset == 'nursery':
        save_nursery_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'adult':
        save_adult_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'german':
        save_german_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif args.dataset == 'mnist':
        save_mnist_party_data(points_per_party, stratify, folder_party_data, folder_dataset, data_distribution)
    elif args.dataset == 'emnist':
        # save_emnist_party_data(points_per_party, stratify, folder_party_data, folder_dataset, data_distribution)
        save_emnist_party_data_with_distributions(points_per_party, stratify, folder_party_data, folder_dataset, data_distribution)
    elif args.dataset == 'compas':
        save_compas_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'higgs':
        save_higgs_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'airline':
        save_airline_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'diabetes':
        save_diabetes_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'binovf':
        save_binovf_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'multovf':
        save_multovf_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'linovf':
        save_linovf_party_data(points_per_party, folder_party_data, folder_dataset)
    elif dataset == 'federated-clustering':
        save_federated_clustering_data(points_per_party, folder_party_data)
    elif dataset == 'femnist':
        save_femnist_party_data(points_per_party, stratify, folder_party_data, folder_dataset)
    elif dataset == 'cifar10':
        save_cifar10_party_data(points_per_party, stratify, folder_party_data, folder_dataset, data_distribution)
    elif dataset == 'wikipedia':
        save_wikipedia_party_data(points_per_party, folder_party_data, folder_dataset)
    else:
        print("Loading a non-default dataset, redircting to general data split method...")
        save_party_data(points_per_party, stratify, folder_party_data, folder_dataset, dataset)
