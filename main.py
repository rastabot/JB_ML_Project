'''
Adult Data set Project
John Bryce --Machine Learning
'''

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from selenium import webdriver

from clean_data import get_clean_df

from train_models import cross_val_train_models

from data_graphs import print_graphs

from split_data import train_test_data


if __name__ == "__main__":
    
    df_train =  train_test_data()[0]
    df_test = train_test_data()[1]

    print('Main Running.......')
   
    # training and cross validation
    cross_val_train_models(df_train)
    
    #running test results 
    #test_results(df_test)   
    
    p = input('press "y" to print graphs \n')
    if p == 'y':
        print_graphs(df_train)
        print('Showing Graphs') 
    
    