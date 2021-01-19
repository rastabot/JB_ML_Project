# -*- coding: utf-8 -*-

import pickle

import pandas as pd

from sklearn.metrics import classification_report,confusion_matrix
                            


def test_results(df):
    
    print('***********TEST RESULTS***********')
    
    models=[]
    mod_name = ['Decision Trees','Random Forest','Neural Network']
      
    
    #print('TEST shape:', df.shape,'\n','TEST Columns:' ,df.columns)
    
    X_test = df.drop('less than 50K',axis=1)
    y_test = df['less than 50K']
    
    #print("X shape = " ,X_test.shape)
    
    # -- loading decision Tree model 
    
    file = open("DT_gs.model",'rb')
    DT_gs = pickle.load(file)
    file.close()
    models.append(DT_gs)
    
    # -- loading Random Forest model    
    
    # RandomForest_gs.model is in a rar ,
    # its around 200mb so it needs to be unzipped
    
    file2 = open("RandomForest_gs.model",'rb')
    rf_gs = pickle.load(file2)
    file2.close()
    models.append(rf_gs)
    
     #-- Loading Neural Network model
    file3 = open("NeuralNetwrok_GsCv.model",'rb')
    clf_NN = pickle.load(file3)
    file3.close()
    models.append(clf_NN)
    
    #model_dict = dict(zip(mod_name,models))
    
    for i,name in enumerate(mod_name):
        print(i+1,name)
    
    for i,name in enumerate(mod_name):
        print('Scores for ',name,' model :',end='\n\n')
        prediction = models[i].predict(X_test)
        score = models[i].score(X_test,y_test)
        print('classification report : ',classification_report(y_test,prediction))
        print(confusion_matrix(y_test,prediction))
        print('Score: ',score)
        print('=========================================','\n\n')
    
    