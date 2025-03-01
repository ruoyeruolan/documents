# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : 002.configuations.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/18 20:58
@Description: 
"""

import optuna
from sklearn import svm
from sklearn import ensemble

import torch
import torch.nn as nn

def objective(trial: optuna.trial.Trial):

    # Categorical parameter
    optimizer = trial.suggest_categorical(name='optimizer', choices=['SGD', 'Adam', 'RMSprop', 'MomenetSGD'])

    # Integer parameter
    num_layers = trial.suggest_int(name='num_layers', low=1, high=3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)
    num_units = trial.suggest_int(name='num_units', low=2, high=8, log=True)

    # Integer discrete parameter
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # float 
    dropout = trial.suggest_float(name='dropout', low=.001, high=.1)


    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)


def objective_branches(trial: optuna.trial.Trial):
    classifier_name = trial.suggest_categorical(name='classifier', choices=['SVM', 'RandomForest'])

    if classifier_name == "SVC":
        svc_c = trial.suggest_float("svc_c", 1e-10, 1e10, log=True)
        classifier_obj = svm.SVC(C=svc_c)
    else:
        rf_max_depth = trial.suggest_int("rf_max_depth", 2, 32, log=True)
        classifier_obj = ensemble.RandomForestClassifier(max_depth=rf_max_depth)

def objective_loop(trial: optuna.trial.Trial, in_size: int):

    n_layers = trial.suggest_int("n_layers", 1, 3)

    layers = []
    for i in range(n_layers):
        n_units = trial.suggest_int(name="n_units_l{}".format(i), low=4, high=128, log=True)
        layers.append(nn.Linear(in_size, n_units))
        layers.append(nn.ReLU())
        in_size = n_units
    layers.append(nn.Linear(in_size, 10))
    return nn.Sequential(*layers)