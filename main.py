'''
Adult Data set Project
John Bryce --Machine Learning
'''

from train_models import cross_val_train_models

from data_graphs import print_graphs

from split_data import train_test_data

from test_and_scores import test_results


if __name__ == "__main__":
    
    #get train set    
    df_train =  train_test_data()[0] 
    
    #get test set
    df_test = train_test_data()[1]

    print('Main Running.......')
   
    # training and cross validation
    cross_val_train_models(df_train)
    
    #running test results 
    test_results(df_test)   
    
    p = input('press "y" to print graphs \n')
    if p == 'y':
        print_graphs()
        print('Showing Graphs') 
    
    