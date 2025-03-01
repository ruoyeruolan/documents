# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : 001_first.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/18 20:45
@Description: 
"""

import optuna

def objective(trial: optuna.trial.Trial):
    x = trial.suggest_float('x', -10, 10)
    return (x - 2) ** 2

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    study.best_params