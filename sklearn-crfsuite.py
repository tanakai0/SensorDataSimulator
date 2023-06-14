#!/usr/bin/env python
# coding: utf-8

import scipy.stats
import numpy as np

import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import scorers, metrics
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import make_scorer

from itertools import chain
from pathlib import Path

# self-made
import activity_model
import analysis
import anomaly
import comparison
import floor_plan

import new_functions
import sensor_model

working_path = Path().resolve()
layout_data_path = working_path / 'layout_data'


def data2features(data):
    """
    Parameters
    ----------
    data : numpy.ndarray
        data.shape = (number of time, number of sensors).
        
    Returns
    -------
    features : list of dict
    """
    
    features = []
    T = data.shape[0]  # number of time
    M = data.shape[1]  # number of sensors
    for i in range(T):
        d = data[i]
        feature = {f"x_{j}": d[j] for j in range(M)}
        # if i >= 1:
        #     feature.update({f"-1 x_{j}": data[i-1][j] for j in range(M)})
        # if i >= 60:
        #     feature['sum_60'] = np.sum(data[i-60:i])
        if i == 0:
            feature['BOS'] = True
        if i == T - 1:
            feature['EOS'] = True
        feature['bias'] = 1
        features.append(feature)
    return features

_type = 'raw'
data_folder_name = 'test_data_1'
path = layout_data_path / 'test_layout' / data_folder_name
reduced_SD_mat = new_functions.pickle_load(path / 'experiment1', f'reduced_SD_mat_{_type}_1')
reduced_AL_mat = new_functions.pickle_load(path / 'experiment1', f'reduced_AL_mat_{_type}_1')
SD_names = new_functions.pickle_load(path / 'experiment1', 'SD_names')
AL_names = new_functions.pickle_load(path / 'experiment1', 'AL_names')


num = 100000000

X_train = [data2features(reduced_SD_mat[-num:, :24])]
y_train = [[str(b) for b in reduced_AL_mat[-num:, 4]]]
print(np.sum(reduced_AL_mat[-num:, 4]))

c1, c2 = 0.1, 0.1
crf = sklearn_crfsuite.CRF(
    algorithm = 'lbfgs', 
    c1 = c1, 
    c2 = c2, 
    max_iterations = 100,
    all_possible_transitions = True
)
crf.fit(X_train, y_train)
new_functions.pickle_dump(path / 'experiment1', f"crf_c1_{c1}_c2_{c2}", crf)

from collections import Counter
from sklearn.metrics import classification_report

# evaluation
_type = 'raw'
data_folder_name = 'test_data_2'
path = layout_data_path / 'test_layout' / data_folder_name
test_SD = new_functions.pickle_load(path / 'experiment1', f'reduced_SD_mat_{_type}_1')
test_AL = new_functions.pickle_load(path / 'experiment1', f'reduced_AL_mat_{_type}_1')
test_SD_names = new_functions.pickle_load(path / f'experiment1', 'SD_names')
test_AL_names = new_functions.pickle_load(path / 'experiment1', 'AL_names')

X_test = [data2features(test_SD[-num:, :24])]
y_test = [[str(b) for b in test_AL[-num:, 4]]]
print(np.sum(test_AL[-num:, 4]))
y_pred = crf.predict(X_test)


labels = list(crf.classes_)
print(labels)
print(metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels))

# details
sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

print(classification_report(
    list(chain.from_iterable(y_test)), list(chain.from_iterable(y_pred)), labels=sorted_labels, digits=3
))


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))
        
def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


print("transition features:")
print_transitions(Counter(crf.transition_features_).most_common())

print("state features:")
print_state_features(Counter(crf.state_features_).most_common())





