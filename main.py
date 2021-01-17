'''
Adult Data set Project
John Bryce --Machine Learning
'''

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from selenium import webdriver

from train_models import cross_val_train_models

from data_graphs import print_graphs







if __name__ == "__main__":
    
    cross_val_train_models()
    
    p = input('press "y" to print graphs \n')
    if p == 'y':
        print_graphs()
        print('Showing Graphs') 
    
    