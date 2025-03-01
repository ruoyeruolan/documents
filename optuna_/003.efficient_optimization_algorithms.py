# -*- encoding: utf-8 -*-
"""
@Introduce  : 
@File       : 003.efficient_optimization_algorithms.py
@Author     : ryrl
@Email      : ryrl970311@gmail.com
@Time       : 2025/02/18 21:27
@Description: 
"""

import optuna

def demo():
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.GPSampler())
    study.sampler.__class__.__name__
