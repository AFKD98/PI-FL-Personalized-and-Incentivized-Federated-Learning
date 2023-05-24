import numpy as np
from sklearn.model_selection import train_test_split

import torchvision
import numpy as np
from sklearn.cluster import KMeans
import torch
import random
from sklearn.model_selection import train_test_split
import os
class EMNIST():

        
    def load_emnist(self):
        """
        Load EMNIST Data from Pytorch.  To simplify data preprocessing, we
        perform this here instead at the DataLoader end to offset the process.
        Preprocessing includes primarily feature dropping.
        returns training, testing dataset
        """
        # additional imports for EMNIST
        import torch
        from torchvision import datasets, transforms

        # Download EMNIST Data
        print('Loading dataset from Pytorch')
        train_data = datasets.EMNIST(root='./data', split='byclass', train=True,
                                    download=True, transform=transforms.ToTensor(), 
                                    target_transform = lambda x: x-10)
        test_data = datasets.EMNIST(root='./data', split='byclass', train=False,
                                    download=True, transform=transforms.ToTensor(), 
                                    target_transform = lambda x: x-10)

        return (train_data.data, train_data.targets), (test_data.data, test_data.targets)
    
    def generate_data(self):

        PIFL = True
        val_split = 0.1
        global_split_train = 0.1
        global_split_test = 0.2
        # Define the number of clusters
        NUM_CLASSES = 52
        NUM_PARTIES = 100
        MAX_CLASSES_PER_PARTY = 4
        n_clusters = NUM_CLASSES//MAX_CLASSES_PER_PARTY
        # Generate sample data with 2 features
        X = np.random.rand(NUM_PARTIES, NUM_PARTIES//n_clusters)
        files_list = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
        print('Number of clusters', n_clusters)


        # Fit the K-means clustering model to the data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Divide the data into clusters, with roughly equal number of clients in each cluster
        cluster_assignments = kmeans.fit_predict(X)
        print('Cluster assignments', cluster_assignments)
        # Calculate the number of clients assigned to each cluster
        cluster_sizes = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_sizes[i] = np.sum(cluster_assignments == i)

        # Adjust the cluster assignments to achieve a more even distribution of clients
        while np.max(cluster_sizes) - np.min(cluster_sizes) > 1:
            # Find the two largest clusters
            largest_cluster = np.argmax(cluster_sizes)
            smallest_cluster = np.argmin(cluster_sizes)
            
            # Find the data point closest to the centroid of the largest cluster
            distances = kmeans.transform(X)[:, largest_cluster]
            closest_point = np.argmin(distances)
            
            # Move the closest point from the largest cluster to the smallest cluster
            cluster_assignments[closest_point] = smallest_cluster
            cluster_sizes[largest_cluster] -= 1
            cluster_sizes[smallest_cluster] += 1

        # Create a dictionary of clusters with client indexes
        clusters_parties = {}
        for i in range(0,n_clusters):
            clusters_parties[i] = np.where(cluster_assignments == i)[0].tolist()

        # Print the number of clients in each cluster
        for i in range(n_clusters):
            print(f"Cluster {i}: {int(cluster_sizes[i])} clients, indexes: {clusters_parties[i]}")





        # Load the EMNIST dataset from PyTorch
        # emnist_dataset_train = torchvision.datasets.EMNIST(root='./data', split='letters', download=True, train=True)
        # emnist_dataset_test = torchvision.datasets.EMNIST(root='./data', split='letters', download=True, train=False)
        (train_data, train_target), (test_data, test_target) = self.load_emnist()
        # Split the dataset by classes
        classes_data = {}
        train_data, test_data = train_data.numpy(), test_data.numpy()
        train_target, test_target = train_target.numpy(), test_target.numpy()
        for x,y in zip(train_data, train_target):
            # print(train_target[0])
            if y>9:
                if y-10 not in classes_data:
                    classes_data[y-10] = {'train': {'data': [], 'target': []}, 'test': {'data': [], 'target': []}}
                classes_data[y-10]['train']['data'].append(x)
                classes_data[y-10]['train']['target'].append(y-10)
            
        for x,y in zip(test_data, test_target):
            if y>9:
                if y-10 not in classes_data:
                    classes_data[y-10] = {'train': {'data': [], 'target': []}, 'test': {'data': [], 'target': []}}
                
                classes_data[y-10]['test']['data'].append(x)
                classes_data[y-10]['test']['target'].append(y-10)
            
        # print('classes_data', len(classes_data))

        # print('emnist_dataset_train', classes_data[1]['test']['target'])
        # Divide the class names into clusters
        clusters_classes = {}
        for i in range(0,n_clusters):
            start_idx = i * MAX_CLASSES_PER_PARTY
            end_idx = (i+1) * MAX_CLASSES_PER_PARTY
            if end_idx > NUM_CLASSES:
                end_idx = NUM_CLASSES
            clusters_classes[np.int64(i)] = np.arange(start_idx, end_idx).tolist()

        # print('clusters_classes', clusters_classes)
        # print(classes_data[0]['test']['target'])
        # print(clusters_classes)
        # Assign data parties to clusters
        for cluster_id, classes in clusters_classes.items():
            print(classes)
            train_dataset_for_cluster = {'x': [], 'y': []}
            test_dataset_for_cluster = {'x': [], 'y': []}
            for class_index in classes:
                # Merge the data parties into a single dataset
                train_data, train_target = classes_data[class_index]['train']['data'], classes_data[class_index]['train']['target']
                test_data, test_target = classes_data[class_index]['test']['data'], classes_data[class_index]['test']['target']
                
                train_dataset_for_cluster['x'] += train_data
                train_dataset_for_cluster['y'] += train_target
                test_dataset_for_cluster['x'] += test_data
                test_dataset_for_cluster['y'] += test_target
            print('len(train dataset_for_cluster)', len(train_dataset_for_cluster['y']))
            print('len(test dataset_for_cluster)', len(test_dataset_for_cluster['y']))
                # Assign data samples to parties
            # Shuffle training data
            train_data = list(zip(train_dataset_for_cluster['x'], train_dataset_for_cluster['y']))
            random.shuffle(train_data)

            train_x, train_y = zip(*train_data)
            train_data = np.array(train_data)
            # Shuffle test data
            test_data = list(zip(test_dataset_for_cluster['x'], test_dataset_for_cluster['y']))
            random.shuffle(test_data)
            test_x, test_y = zip(*test_data)
            test_data = np.array(test_data)
            
            train_x, train_y = np.array(train_x), np.array(train_y)
            #Save the tier level global dataset
            DA_train_ids = np.random.choice(range(len(train_data)), size=int(len(train_data)*global_split_train), replace=False)
            DA_test_ids = np.random.choice(range(len(test_data)), size=int(len(test_data)*global_split_test), replace=False)
            
            # print('DA_test_x_ids', DA_test_x_ids)
            DA_train_x, DA_train_y = zip(*train_data[DA_train_ids])
            DA_test_x, DA_test_y = zip(*test_data[DA_test_ids])
            
            # DA_train = np.random.choice(train_data, size=int(len(train_data)*global_split_train), replace=False)
            # DA_test = np.random.choice(test_data, size=int(len(test_data)*global_split_test), replace=False)
            
            # DA_train_x, DA_train_y = zip(*DA_train)
            # DA_test_x, DA_test_y = zip(*DA_test)
            
            np.savez('data_parties/emnist_D{}.npz'.format(files_list[cluster_id]), 
            x_train=np.array(DA_train_x), y_train=np.array(DA_train_y), 
            x_test=np.array(DA_test_x), y_test=np.array(DA_test_y))

            #print labels owned by each cluster
            
            print(f"Cluster {cluster_id} has {classes} labels")
            i = 0
            parties_in_cluster = len(clusters_parties[cluster_id])
            for party_id in clusters_parties[cluster_id]:
                start_train = i * len(train_y) // parties_in_cluster
                end_train = (i+1) * len(train_y) // parties_in_cluster
                start_test = i * len(test_y) // parties_in_cluster
                end_test = (i+1) * len(test_y) // parties_in_cluster
                print(f"Party {party_id} has {end_train-start_train} samples")
                train_dataset = {'data': {'x': np.array(train_x[start_train:end_train]), 'y': np.array(train_y[start_train:end_train])}}
                test_dataset = {'data': {'x': np.array(test_x[start_test:end_test]), 'y': np.array(test_y[start_test:end_test])}}
                
                
                if PIFL:
                    
                    party_x_train = np.array(train_x[start_test:end_test])
                    party_y_train = np.array(train_y[start_test:end_test])
                    party_train_ids = np.random.choice(range(len(party_x_train)), size=int(len(party_x_train)*val_split), replace=False)
                    party_x_val, party_y_val = party_x_train[party_train_ids], party_y_train[party_train_ids]
                    print('party_x_train', party_x_train.shape)
                    party_x_train, party_y_train = np.delete(party_x_train, party_train_ids, axis=0).reshape((-1, 28, 28, 1)), np.delete(party_y_train, party_train_ids)
                    
                    np.savez(f'data_parties/data_party{party_id}.npz', x_train=party_x_train, y_train=party_y_train,
                            x_val=party_x_val, y_val=party_y_val,
                            x_test=np.array(test_x[start_test:end_test]), y_test=np.array(test_y[start_test:end_test]))
                else:
                    np.savez(f'mnist/train/{party_id}.npz', data={'x': np.array(train_x[start_train:end_train]), 'y': np.array(train_y[start_train:end_train])})
                    np.savez(f'mnist/test/{party_id}.npz', data={'x': np.array(test_x[start_test:end_test]), 'y': np.array(test_y[start_test:end_test])})
                # np.savez(f'mnist/train/{party_id}.npz', x_train=train_dataset['data']['x'], y_train=train_dataset['data']['y'])
                # np.savez(f'mnist/test/{party_id}.npz', x_train=test_dataset['data']['x'], y_train=test_dataset['data']['y'])
                i+=1
                
                
class CIFAR10():

        
    def load_cifar10(self):
        """
        Load EMNIST Data from Pytorch.  To simplify data preprocessing, we
        perform this here instead at the DataLoader end to offset the process.
        Preprocessing includes primarily feature dropping.
        returns training, testing dataset
        """
        # additional imports for EMNIST
        import torch
        from torchvision import datasets, transforms

        # Download EMNIST Data
        print('Loading dataset from Pytorch')
        train_data = datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transforms.ToTensor())
        test_data = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=transforms.ToTensor())

        return (train_data.data, train_data.targets), (test_data.data, test_data.targets)
    
    def generate_data(self):

        PIFL = True
        (train_data, train_target), (test_data, test_target) = self.load_cifar10()
        val_split = 0.1
        global_split_train = 0.1
        global_split_test = 0.2
        # Define the number of clusters
        NUM_CLASSES = 10
        print('Number of classes', NUM_CLASSES)
        NUM_PARTIES = 100
        MAX_CLASSES_PER_PARTY = 2
        n_clusters = NUM_CLASSES//MAX_CLASSES_PER_PARTY
        # Generate sample data with 2 features
        X = np.random.rand(NUM_PARTIES, NUM_PARTIES//n_clusters)
        files_list = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
        print('Number of clusters', n_clusters)


        # Fit the K-means clustering model to the data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Divide the data into clusters, with roughly equal number of clients in each cluster
        cluster_assignments = kmeans.fit_predict(X)
        print('Cluster assignments', cluster_assignments)
        # Calculate the number of clients assigned to each cluster
        cluster_sizes = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_sizes[i] = np.sum(cluster_assignments == i)

        # Adjust the cluster assignments to achieve a more even distribution of clients
        while np.max(cluster_sizes) - np.min(cluster_sizes) > 1:
            # Find the two largest clusters
            largest_cluster = np.argmax(cluster_sizes)
            smallest_cluster = np.argmin(cluster_sizes)
            
            # Find the data point closest to the centroid of the largest cluster
            distances = kmeans.transform(X)[:, largest_cluster]
            closest_point = np.argmin(distances)
            
            # Move the closest point from the largest cluster to the smallest cluster
            cluster_assignments[closest_point] = smallest_cluster
            cluster_sizes[largest_cluster] -= 1
            cluster_sizes[smallest_cluster] += 1

        # Create a dictionary of clusters with client indexes
        clusters_parties = {}
        for i in range(0,n_clusters):
            clusters_parties[i] = np.where(cluster_assignments == i)[0].tolist()

        # Print the number of clients in each cluster
        for i in range(n_clusters):
            print(f"Cluster {i}: {int(cluster_sizes[i])} clients, indexes: {clusters_parties[i]}")





        # Load the EMNIST dataset from PyTorch
        # cifar10_dataset_train = torchvision.datasets.EMNIST(root='./data', split='letters', download=True, train=True)
        # cifar10_dataset_test = torchvision.datasets.EMNIST(root='./data', split='letters', download=True, train=False)
        # Split the dataset by classes
        classes_data = {}
        # train_data, test_data = train_data.numpy(), test_data.numpy()
        # train_target, test_target = train_target.numpy(), test_target.numpy()
        for x,y in zip(train_data, train_target):
            # print(train_target[0])
                if y not in classes_data:
                    classes_data[y] = {'train': {'data': [], 'target': []}, 'test': {'data': [], 'target': []}}
                classes_data[y]['train']['data'].append(x)
                classes_data[y]['train']['target'].append(y)
            
        for x,y in zip(test_data, test_target):
                if y not in classes_data:
                    classes_data[y] = {'train': {'data': [], 'target': []}, 'test': {'data': [], 'target': []}}
                
                classes_data[y]['test']['data'].append(x)
                classes_data[y]['test']['target'].append(y)
            
        # print('classes_data', len(classes_data))

        # print('cifar10_dataset_train', classes_data[1]['test']['target'])
        # Divide the class names into clusters
        clusters_classes = {}
        for i in range(0,n_clusters):
            start_idx = i * MAX_CLASSES_PER_PARTY
            end_idx = (i+1) * MAX_CLASSES_PER_PARTY
            if end_idx > NUM_CLASSES:
                end_idx = NUM_CLASSES
            clusters_classes[np.int64(i)] = np.arange(start_idx, end_idx).tolist()

        # print('clusters_classes', clusters_classes)
        # print(classes_data[0]['test']['target'])
        # print(clusters_classes)
        # Assign data parties to clusters
        for cluster_id, classes in clusters_classes.items():
            print(classes)
            train_dataset_for_cluster = {'x': [], 'y': []}
            test_dataset_for_cluster = {'x': [], 'y': []}
            for class_index in classes:
                # Merge the data parties into a single dataset
                train_data, train_target = classes_data[class_index]['train']['data'], classes_data[class_index]['train']['target']
                test_data, test_target = classes_data[class_index]['test']['data'], classes_data[class_index]['test']['target']
                
                train_dataset_for_cluster['x'] += train_data
                train_dataset_for_cluster['y'] += train_target
                test_dataset_for_cluster['x'] += test_data
                test_dataset_for_cluster['y'] += test_target
            print('len(train dataset_for_cluster)', len(train_dataset_for_cluster['y']))
            print('len(test dataset_for_cluster)', len(test_dataset_for_cluster['y']))
                # Assign data samples to parties
            # Shuffle training data
            train_data = list(zip(train_dataset_for_cluster['x'], train_dataset_for_cluster['y']))
            random.shuffle(train_data)

            train_x, train_y = zip(*train_data)
            train_data = np.array(train_data)
            # Shuffle test data
            test_data = list(zip(test_dataset_for_cluster['x'], test_dataset_for_cluster['y']))
            random.shuffle(test_data)
            test_x, test_y = zip(*test_data)
            test_data = np.array(test_data)
            
            train_x, train_y = np.array(train_x), np.array(train_y)
            #Save the tier level global dataset
            DA_train_ids = np.random.choice(range(len(train_data)), size=int(len(train_data)*global_split_train), replace=False)
            DA_test_ids = np.random.choice(range(len(test_data)), size=int(len(test_data)*global_split_test), replace=False)
            
            # print('DA_test_x_ids', DA_test_x_ids)
            DA_train_x, DA_train_y = zip(*train_data[DA_train_ids])
            DA_test_x, DA_test_y = zip(*test_data[DA_test_ids])
            
            # DA_train = np.random.choice(train_data, size=int(len(train_data)*global_split_train), replace=False)
            # DA_test = np.random.choice(test_data, size=int(len(test_data)*global_split_test), replace=False)
            
            # DA_train_x, DA_train_y = zip(*DA_train)
            # DA_test_x, DA_test_y = zip(*DA_test)
            if not os.path.exists(f'data_parties/'):
                os.makedirs(f'data_parties/')
                        
            np.savez('data_parties/cifar10_D{}.npz'.format(files_list[cluster_id]), 
            x_train=np.array(DA_train_x), y_train=np.array(DA_train_y), 
            x_test=np.array(DA_test_x), y_test=np.array(DA_test_y))

            #print labels owned by each cluster
            
            print(f"Cluster {cluster_id} has {classes} labels")
            i = 0
            parties_in_cluster = len(clusters_parties[cluster_id])
            for party_id in clusters_parties[cluster_id]:
                start_train = i * len(train_y) // parties_in_cluster
                end_train = (i+1) * len(train_y) // parties_in_cluster
                start_test = i * len(test_y) // parties_in_cluster
                end_test = (i+1) * len(test_y) // parties_in_cluster
                print(f"Party {party_id} has {end_train-start_train} samples")
                train_dataset = {'data': {'x': np.array(train_x[start_train:end_train]), 'y': np.array(train_y[start_train:end_train])}}
                test_dataset = {'data': {'x': np.array(test_x[start_test:end_test]), 'y': np.array(test_y[start_test:end_test])}}
                
                
                if PIFL:
                    
                    party_x_train = np.array(train_x[start_test:end_test])
                    party_y_train = np.array(train_y[start_test:end_test])
                    party_train_ids = np.random.choice(range(len(party_x_train)), size=int(len(party_x_train)*val_split), replace=False)
                    party_x_val, party_y_val = party_x_train[party_train_ids], party_y_train[party_train_ids]
                    print('party_x_train', party_x_train.shape)
                    party_x_train, party_y_train = np.delete(party_x_train, party_train_ids, axis=0).reshape((-1, 32, 32, 3)), np.delete(party_y_train, party_train_ids)
                    
                    
                    np.savez(f'data_parties/data_party{party_id}.npz', x_train=party_x_train, y_train=party_y_train,
                            x_val=party_x_val, y_val=party_y_val,
                            x_test=np.array(test_x[start_test:end_test]), y_test=np.array(test_y[start_test:end_test]))
                else:
                    np.savez(f'mnist/train/{party_id}.npz', data={'x': np.array(train_x[start_train:end_train]), 'y': np.array(train_y[start_train:end_train])})
                    np.savez(f'mnist/test/{party_id}.npz', data={'x': np.array(test_x[start_test:end_test]), 'y': np.array(test_y[start_test:end_test])})
                # np.savez(f'mnist/train/{party_id}.npz', x_train=train_dataset['data']['x'], y_train=train_dataset['data']['y'])
                # np.savez(f'mnist/test/{party_id}.npz', x_train=test_dataset['data']['x'], y_train=test_dataset['data']['y'])
                i+=1


if __name__ == '__main__':
    cifar10 = CIFAR10()
    cifar10.generate_data()
    