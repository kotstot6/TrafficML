
from road_network import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random
import math
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

torch.random.manual_seed(1)
random.seed(1)
np.random.seed(1)

class Dataset:

    def __init__(self, data, K=10):

        self.K = K
        self.tt_splits = self.cross_val_splits(data, K)
        self.info = {}
        self.data = data

    def cross_val_splits(self, data, K):

        random.shuffle(data)

        test_size = len(data) // K

        test_sets = [data[test_size*i:test_size*(i+1)] for i in range(K)]
        train_sets = [data[:test_size*i] + data[test_size*(i+1):] for i in range(K)]

        return list(zip(train_sets, test_sets))

    def normalize(self, column):

        if column not in self.info:
            self.info[column] = {}

        self.info[column]['norm'] = []

        for train_set, _ in self.tt_splits:
            data = np.array([row[column] for row, _ in train_set if row[column] is not None])
            self.info[column]['norm'].append((np.mean(data), np.std(data)))

    def cluster(self, column, K):

        if column not in self.info:
            self.info[column] = {}

        self.info[column]['cluster'] = []

        for train_set, _ in self.tt_splits:
            data = np.array([np.array(row[column]) for row, _ in train_set if row[column] is not None])
            kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
            self.info[column]['cluster'].append(kmeans)

    def fill_missing(self, column):

        if column not in self.info:
            self.info[column] = {}

        self.info[column]['fill'] = []

        for train_set, _ in self.tt_splits:
            try:
                data = np.array([float(row[column]) for row, _ in train_set if row[column] is not None])
                self.info[column]['fill'].append(np.mean(data))
            except:
                self.info[column]['fill'].append(0)

    def encode_labels(self, column):

        if column not in self.info:
            self.info[column] = {}

        data = []

        for train_set, _ in self.tt_splits:
            data += [str(row[column]) for row, _ in train_set]

        self.info[column]['encode'] = sorted(list(set(data)))

    def encode_feature(self, feature, i):


        X = []

        for col, val in feature.items():

            if col not in self.info:
                X.append(val)
                continue

            # possible combos:
            # fill, fill-norm, fill-cluster, norm, cluster, encode

            if 'fill' in self.info[col] and 'norm' in self.info[col]:
                mean, std = self.info[col]['norm'][i]
                X += [1, 0] if val is None else [0, (val - mean) / std]

            elif 'fill' in self.info[col] and 'cluster' in self.info[col]:
                kmeans = self.info[col]['cluster'][i]
                N = kmeans.n_clusters
                if val is None:
                    X += [1] + [0] * N
                else:
                    label = kmeans.predict([val])[0]
                    X += [0] + [int(j == label) for j in range(N)]

            elif 'fill' in self.info[col]:
                mean = self.info[col]['fill'][i]
                X += [1, mean] if val is None else [0, val]

            elif 'norm' in self.info[col]:
                mean, std = self.info[col]['norm'][i]
                X.append((val - mean) / std)

            elif 'cluster' in self.info[col]:
                kmeans = self.info[col]['cluster'][i]
                label = kmeans.predict([val])[0]
                X += [int(j == label) for j in range(kmeans.n_clusters)]

            elif 'encode' in self.info[col]:
                labels = self.info[col]['encode']
                X += [int(val == l) for l in labels]

            else:
                X.append(val)

        try:
            return torch.Tensor(X)
        except:
            return torch.zeros(1)

    def normalize_targets(self):

        self.info['Target'] = []

        for train_set, _ in self.tt_splits:

            data = torch.Tensor([target for _, target in train_set])
            self.info['Target'].append((torch.mean(data, dim=0), torch.std(data, dim=0)))

    def encode_target(self, target, i):

        return torch.Tensor(target)

    def preprocess(self):

        self.tt_data = []

        for i, (train_set, test_set) in enumerate(self.tt_splits):
            train_data = [(self.encode_feature(X,i), self.encode_target(Y,i)) for X,Y in train_set]
            test_data = [(self.encode_feature(X,i), self.encode_target(Y,i)) for X,Y in test_set]
            self.tt_data.append((train_data, test_data))

    def feature_length(self):
        return len(self.tt_data[0][0][0][0])

    def target_length(self):
        return len(self.tt_data[0][0][0][1])

    def train_batches(self, i, size=-1):

        train_set, _ = self.tt_data[i]
        random.shuffle(train_set)
        size = len(train_set) if size < 0 else size
        return [self.make_batch(train_set[size*j:size*(j+1)], i) for j in range(math.ceil(len(train_set) / size))]

    def test_batch(self, i):
        _, test_set = self.tt_data[i]
        return self.make_batch(test_set, i)

    def make_batch(self, data, i):

        M = len(data)
        X = torch.zeros((M, len(data[0][0])))
        Y = torch.zeros((M, len(data[0][1])))

        for j in range(M):
            x, y = data[j]
            X[j,:] = x
            Y[j,:] = y

        return X,Y

    def baseline_loss(self, type, index=0):

        assert type in ['mean', 'median']

        _ , Y_train = self.train_batches(index)[0]
        _ , Y_test = self.test_batch(index)

        Y_predict = torch.mean(Y_train, dim=0) if type =='mean' else torch.median(Y_train, dim=0).values

        return float(torch.mean(torch.abs(Y_test - Y_predict)))

    def feature_matrix(self):

        data = [x for x, _ in self.data]

        features  = [self.encode_feature(x, 0) for x in data]

        m, n = len(data), len(features[0])

        X = torch.zeros((m,n))

        for i in range(m):
            X[i,:] = features[i]

        return X


class KNearestNeighbors:

    def __init__(self, road_net):

        self.road_net = road_net

        self.dataset = Dataset([ (self.make_feature(station.edge), self.make_target(station))
                            for station in road_net.stations])

        self.dataset.preprocess()

    def make_feature(self, edge):

        return {
            'ID-1' : edge.nodes[0].id,
            'ID-2' : edge.nodes[0].id
            }

    def make_target(self, station):
        return station.volume

    def predict_by_node(self, node_id, K=1, index=0):

        train_set, _ = self.dataset.tt_splits[index]
        train_nodes = { x['ID-1'] : y for x,y in train_set }

        knn_nodes = self.get_KNN(K, node_id, list(train_nodes.keys()))

        return np.median(np.array([train_nodes[node] for node in knn_nodes]), axis=0)

    def predict(self, edges, K=1, index=0):

        vols = []

        for i, edge in enumerate(edges):
            print(i, '/', len(edges))
            vol = self.predict_by_node(edge.nodes[1].id, K=K, index=index)
            vols.append([round(v) for v in list(vol)])

        return vols


    def test(self, K=1, index=0, verbose=True):

        if verbose:
            print('Testing...')

        baseline_loss = self.dataset.baseline_loss('median', index=index)

        if verbose:
            print('Baseline loss:', baseline_loss)

        train_set, test_set = self.dataset.tt_splits[index]

        sets = {'Train' : train_set, 'Test' : test_set}

        loss = {}

        for type, data in sets.items():

            nodes = { x['ID-2'] : np.array(y) for x,y in data }

            losses = []

            for node_id, vol in nodes.items():
                vol_hat = self.predict_by_node(node_id, K=K, index=index)
                losses.append(np.mean(np.abs(vol_hat - vol)))

            loss[type] = np.mean(np.array(losses))

            if verbose:
                print(type, 'loss:', loss[type])

        return baseline_loss, loss['Train'], loss['Test']


    def get_KNN(self, K, node_id, neighbors):

        lengths, _ = nx.single_source_dijkstra(self.road_net.nx_graph, node_id, weight=lambda n1,n2,d: float(d['length']))

        neighbor_lengths = [(id, l) for id, l in lengths.items() if id in neighbors]
        neighbor_lengths += [(n, 1e10) for n in neighbors]
        neighbor_lengths.sort(key=lambda x: x[1])

        return [id for id, _ in neighbor_lengths[:K]]

    def validation_curve(self, K_range, name):

        losses = {
                    'train' : {'mean' : [], 'std' : []},
                    'test' : {'mean' : [], 'std' : []}
                    }

        for K in K_range:

            K = int(K)

            print()
            print('K = ', K)
            print('----------')

            baseline_losses = []
            train_losses = []
            test_losses = []

            for i in range(10):
                bl, trl, tel = self.test(K=K, index=i)
                baseline_losses.append(bl)
                train_losses.append(trl)
                test_losses.append(tel)

            losses['train']['mean'].append(np.mean(train_losses))
            losses['train']['std'].append(np.std(train_losses))
            losses['test']['mean'].append(np.mean(test_losses))
            losses['test']['std'].append(np.std(test_losses))

        result_dict = {
            'baseline' : np.mean(baseline_losses),
            'K_range' : list(K_range),
            'losses' : losses
        }

        with open('results/' + name, 'w') as f:
            f.write(str(result_dict))


def data_preprocess(dataset, type):

    assert type in ['simple', 'complex']

    if type == 'simple':

        dataset.cluster('Location', 20)
        dataset.normalize('Distance')
        dataset.fill_missing('Speed')
        dataset.normalize('Speed')
        dataset.fill_missing('Lanes')
        dataset.normalize('Lanes')
        dataset.encode_labels('E/W')
        dataset.encode_labels('N/S')
        dataset.normalize('Speed/Dist')
        dataset.normalize('Dist*Lanes')
        dataset.normalize('In Degree')
        dataset.normalize('Out Degree')
        dataset.normalize('Betweenness')
        dataset.normalize('Closeness')
        dataset.preprocess()

    else:

        labels = ['*', 'F1', 'L1', 'R1', 'F2', 'L2', 'R2', 'B']

        for L in labels[:1]:
            dataset.fill_missing('Location-' + L)
            dataset.cluster('Location-' + L, 20)
            dataset.fill_missing('Distance-' + L)
            dataset.normalize('Distance-' + L)
            dataset.fill_missing('Speed-' + L)
            dataset.normalize('Speed-' + L)
            dataset.fill_missing('Lanes-' + L)
            dataset.normalize('Lanes-' + L)
            dataset.encode_labels('E/W-' + L)
            dataset.encode_labels('N/S-' + L)
            dataset.fill_missing('Betweenness-' + L)
            dataset.normalize('Betweenness-' + L)
            dataset.fill_missing('Closeness-' + L)
            dataset.normalize('Closeness-' + L)
            dataset.fill_missing('Arctan-' + L)
            dataset.fill_missing('Speed/Dist-' + L)
            dataset.normalize('Speed/Dist-' + L)
            dataset.fill_missing('Dist*Lanes-' + L)
            dataset.normalize('Dist*Lanes-' + L)
            dataset.fill_missing('In Degree-' + L)
            dataset.normalize('In Degree-' + L)
            dataset.fill_missing('Out Degree-' + L)
            dataset.normalize('Out Degree-' + L)

        dataset.preprocess()

def data_make_feature(edge, type):

    assert type in ['simple', 'complex']

    if type == 'simple':
        return {
            'Location' : edge.location(),
            'Distance' : edge.distance,
            'Speed' : edge.speed_limit,
            'Lanes' : edge.lanes,
            'E/W' : edge.direction[0],
            'N/S' : edge.direction[1],
            'Arctan' : edge.arctan(),
            'Speed/Dist' : edge.speed_div_dist(),
            'Dist*Lanes' : edge.dist_times_lanes(),
            'In Degree' : edge.in_deg,
            'Out Degree' : edge.out_deg,
            'Betweenness' : edge.betweenness,
            'Closeness' : edge.nodes[0].closeness
            }
    else:

        feature_dict = {}

        zip_items = [('*', edge)] + list(edge.adjacency.items())

        for L, E in zip_items[:1]:

            if E is not None:
                feature_dict.update({
                    'Distance-' + L : E.distance,
                    'Location-' + L : E.location(),
                    'Speed-' + L : E.speed_limit,
                    'Lanes-' + L : E.lanes,
                    'E/W-' + L : E.direction[0],
                    'N/S-' + L : E.direction[1],
                    'Betweenness-' + L : E.betweenness,
                    'Closeness-' + L : E.nodes[0].closeness,
                    'Arctan-' + L : E.arctan(),
                    'Speed/Dist-' + L : E.speed_div_dist(),
                    'Dist*Lanes-' + L : E.dist_times_lanes(),
                    'In Degree-' + L : E.in_deg,
                    'Out Degree-' + L : E.out_deg
                })
            else:
                feature_dict.update({
                    'Distance-' + L : None,
                    'Location-' + L : None,
                    'Speed-' + L : None,
                    'Lanes-' + L : None,
                    'E/W-' + L : None,
                    'N/S-' + L : None,
                    'Betweenness-' + L : None,
                    'Closeness-' + L : None,
                    'Arctan-' + L : None,
                    'Speed/Dist-' + L : None,
                    'Dist*Lanes-' + L : None,
                    'In Degree-' + L : None,
                    'Out Degree-' + L : None
                })

        return feature_dict


class DecisionTree:

    def __init__(self, road_net, type):

        assert type in ['simple', 'complex']

        if type == 'complex':
            road_net.compute_edge_adjacency()

        self.rn = road_net
        self.type = type

        self.dataset = Dataset([ (self.make_feature(station.edge, type), self.make_target(station))
                            for station in road_net.stations])

        data_preprocess(self.dataset, type)

    def make_feature(self, edge, type):
        return data_make_feature(edge, type)

    def make_target(self, station):
        return station.volume

    def train(self, ensemble=False,
                    max_depth=None,
                    splitter='random',
                    max_features=None,
                    n_estimators=100,
                    index=0):

        if ensemble:
            self.model = RandomForestRegressor(criterion='mae',
                            max_depth=max_depth, n_estimators=n_estimators, random_state=1)
        else:
            self.model = DecisionTreeRegressor(criterion='mae',
                                            max_depth=max_depth,
                                            splitter=splitter,
                                            max_features=max_features,
                                            random_state=1)

        X_train, y_train = self.dataset.train_batches(index)[0]

        self.model.fit(X_train.numpy(), y_train.numpy())

        print('Testing...')

        baseline_loss = self.dataset.baseline_loss('median', index=index)
        print('Baseline Loss:', baseline_loss)

        y_pred = self.model.predict(X_train.numpy())
        train_loss = np.mean(np.abs(y_train.numpy() - y_pred))
        print('Train Loss:', train_loss)

        X_test, y_test = self.dataset.test_batch(index)
        y_pred = self.model.predict(X_test.numpy())
        test_loss = np.mean(np.abs(y_test.numpy() - y_pred))
        print('Test Loss:', test_loss)

        return baseline_loss, train_loss, test_loss

    def predict(self, edges):

        dat_info = self.dataset.info

        dataset = Dataset([(self.make_feature(edge,self.type), [0]) for edge in edges])

        dataset.info = dat_info

        X = dataset.feature_matrix().numpy()

        y = self.model.predict(X)

        volumes = []

        for i in range(y.shape[0]):
            y_i = list(y[i,:])
            volumes.append([round(vol) for vol in y_i])

        return volumes

    def validation_curve(self, param_ranges, ensemble, name):

        param1, param2 = tuple(param_ranges.keys())
        p1_range, p2_range = tuple(param_ranges.values())

        losses = [{
                    'train' : {'mean' : [], 'std' : []},
                    'test' : {'mean' : [], 'std' : []}
                    } for p1 in p1_range]

        for i, p1 in enumerate(p1_range):

            for p2 in p2_range:

                p1, p2 = p1, int(p2)

                print()
                print(param1, ',', param2, '=', p1, ',', p2)
                print('----------')

                baseline_losses = []
                train_losses = []
                test_losses = []

                max_depth = (p1 if param1 == 'max_depth'
                                else p2 if param2 == 'max_depth'
                                    else None)

                max_features = (p1 if param1 == 'max_features'
                                else p2 if param2 == 'max_features'
                                    else 'best')

                n_estimators = (p1 if param1 == 'n_estimators'
                                else p2 if param2 == 'n_estimators'
                                    else 1)

                for j in range(10):

                    bl, trl, tel = self.train(
                                    ensemble=ensemble,
                                    max_depth=max_depth,
                                    max_features=max_features,
                                    n_estimators=n_estimators,
                                    index=j)
                    baseline_losses.append(bl)
                    train_losses.append(trl)
                    test_losses.append(tel)

                losses[i]['train']['mean'].append(np.mean(train_losses))
                losses[i]['train']['std'].append(np.std(train_losses))
                losses[i]['test']['mean'].append(np.mean(test_losses))
                losses[i]['test']['std'].append(np.std(test_losses))

        result_dict = {
            'baseline' : np.mean(baseline_losses),
            param1 : list(p1_range),
            param2 : list(p2_range),
            'losses' : losses
        }

        with open('results/' + name, 'w') as f:
            f.write(str(result_dict))

class Model(nn.Module):

    def __init__(self, input_len, output_len, hidden_layers):

        super(Model, self).__init__()

        layer_sizes = [input_len]
        for i in range(1, hidden_layers+1):
            layer_sizes.append(round(input_len + (i / (hidden_layers+1)) * (output_len - input_len)))
        layer_sizes.append(output_len)

        self.fc = nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.fc.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, x):

        for fc in self.fc[:-1]:
            x = F.relu(fc(x))

        return self.fc[-1](x)


class NeuralNet:

    def __init__(self, road_net, type):

        assert type in ['simple', 'complex']

        if type == 'complex':
            road_net.compute_edge_adjacency()

        self.type = type
        self.rn = road_net
        self.dataset = Dataset([ (self.make_feature(station.edge, type), self.make_target(station))
                            for station in road_net.stations])

        data_preprocess(self.dataset, type)


    def make_feature(self, edge, type):
        return data_make_feature(edge, type)

    def make_target(self, station):
        return station.volume

    def train(self, epochs=10000, test=True, out_freq=100,
                lr=1e-4, hidden_layers=15, early_stopping=True, counter_limit=5, verbose=True, index=0):

        counter_limit = counter_limit if early_stopping else 10000

        self.model = Model(self.dataset.feature_length(), self.dataset.target_length(), hidden_layers)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()

        results = {
            'baseline_loss' : None,
            'train_loss' : None,
            'test_loss' : None,
            'epochs' : [],
            'train_history' : [],
            'test_history' : []
        }

        counter = 0

        for i in range(1, epochs+1):

            losses = []

            batches = self.dataset.train_batches(index, size=32)

            for X,Y in batches:

                optimizer.zero_grad()
                Y_hat = self.model(X)
                loss = torch.mean(torch.abs(Y - Y_hat))
                loss.backward()
                losses.append(float(loss))
                optimizer.step()

            if i % out_freq == 0:

                train_loss = float(torch.mean(torch.Tensor(losses)))

                if verbose:
                    print('Epoch', i)
                    print('---------')
                    print('Train loss:', train_loss)

                if test:
                    baseline_loss, test_loss = self.test(index, verbose)

                    if results['test_loss'] is None or test_loss <= results['test_loss']:

                        results['test_loss'] = test_loss
                        results['train_loss'] = train_loss
                        results['baseline_loss'] = baseline_loss

                        if early_stopping:
                            torch.save(self.model.state_dict(), 'model.dat')

                        counter = 0
                    else:
                        counter += 1

                    results['epochs'].append(i)
                    results['train_history'].append(train_loss)
                    results['test_history'].append(test_loss)

                    if counter >= counter_limit:
                        break

                    self.model.train()

                if verbose:
                    print('---------')

        if early_stopping:
            self.model = Model(self.dataset.feature_length(), self.dataset.target_length(), hidden_layers)
            self.model.load_state_dict(torch.load('model.dat'))

        print('Baseline loss:', results['baseline_loss'])
        print('Train loss:', results['train_loss'])
        print('Test loss:', results['test_loss'])
        print()

        return results

    def test(self, index, verbose):
        baseline_loss = self.dataset.baseline_loss('median', index=index)

        if verbose:
            print('Baseline loss:', baseline_loss)

        self.model.eval()
        with torch.no_grad():
            X, Y = self.dataset.test_batch(index)
            Y_hat = self.model(X)
            test_loss = float(torch.mean(torch.abs(Y - Y_hat)))

            if verbose:
                print('Test loss:', test_loss)

        return baseline_loss, test_loss

    def predict(self, edges):

        dat_info = self.dataset.info

        dataset = Dataset([(self.make_feature(edge,self.type), [0]) for edge in edges])

        dataset.info = dat_info

        X = dataset.feature_matrix()

        self.model.eval()

        with torch.no_grad():
            y = self.model(X).detach().numpy()

        volumes = []

        for i in range(y.shape[0]):
            y_i = list(y[i,:])
            volumes.append([round(vol) for vol in y_i])

        return volumes


    def validation_curve(self, lr_range, hl_range, name):

        losses = [{
                    'train' : {'mean' : [], 'std' : []},
                    'test' : {'mean' : [], 'std' : []}
                    } for lr in lr_range]

        for i, lr in enumerate(lr_range):

            for hl in hl_range:

                lr, hl = lr, int(hl)

                print()
                print('lr , hl =', lr, ',', hl)
                print('----------')

                baseline_losses = []
                train_losses = []
                test_losses = []

                for j in range(10):

                    nn_results = self.train(index=j,
                                lr=lr, hidden_layers=hl, verbose=False)
                    baseline_losses.append(nn_results['baseline_loss'])
                    train_losses.append(nn_results['train_loss'])
                    test_losses.append(nn_results['test_loss'])

                losses[i]['train']['mean'].append(np.mean(train_losses))
                losses[i]['train']['std'].append(np.std(train_losses))
                losses[i]['test']['mean'].append(np.mean(test_losses))
                losses[i]['test']['std'].append(np.std(test_losses))

        result_dict = {
            'baseline' : np.mean(baseline_losses),
            'lr' : list(lr_range),
            'hidden_layers' : list(hl_range),
            'losses' : losses
        }

        with open('results/' + name, 'w') as f:
            f.write(str(result_dict))

    def test_performance(self, index):

        self.model.eval()

        with torch.no_grad():
            X_test, y_test = self.dataset.test_batch(index)
            y_pred = self.model(X_test).detach().numpy()

        loss_data = list(np.abs(y_test.numpy() - y_pred).reshape(-1))

        return loss_data
