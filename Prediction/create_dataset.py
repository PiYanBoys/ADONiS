#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 20:10:18 2022

@author: roy
"""

import pandas as pd

def divide_dataset():  
    a=pd.read_csv("timeser/lorenz.dat", header=None)[0][:1500]
    a.to_csv("timeser/lorenz_train.csv",header=False)
    a=pd.read_csv("timeser/lorenz.dat", header=None)[0][1500:1700]  
    a.to_csv("timeser/lorenz_test.csv", header=False)
    
divide_dataset()