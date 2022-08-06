#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:10:18 2022

@author: roy
"""

import pandas as pd

def divide_dataset():  
    a=pd.read_csv("timeser/lorenz.dat", header=None)[0][:700]
    a.to_csv("timeser/lorenz_train.csv",header=False)
    a=pd.read_csv("timeser/lorenz.dat", header=None)[0][691:1000]  
    a.to_csv("timeser/lorenz_test.csv", header=False)