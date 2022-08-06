#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:41:26 2022

@author: roy
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

nf_train_set = pd.read_csv("data/Lorenz/noise_free_test.csv", header=None)[0]
index=[]
count=0
for i in nf_train_set:
    index += [count]
    count += 1
plt.scatter(index,nf_train_set)
plt.show()