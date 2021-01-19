'''
 cross-validation  and traning 3 models:
    dessision trees, Random Forest and Neural Networks'''


import pickle
    

#from clean_data import get_clean_df


def cross_val_train_models(df_train):
    
    print("Cross validation and training the models, showing best parameters")
    print("-----------------------------------------------------")
    print(" ")  
   
    
    #print(df.columns)
    
    # --- ----------------
    #X = df.drop('less than 50K',axis=1)
    #y = df['less than 50K']
    
    
    # ---------------decision tree with grid search------------
    '''
    tree_params = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
    clf_dt = GridSearchCV(DecisionTreeClassifier(), tree_params,verbose=1, cv=5)
    clf_dt.fit(X_train, y_train)
    '''
    #-- pickling the DT_gs model
    '''
    filehandler1 = open("DT_gs.model","wb")
    pickle.dump(clf_dt,filehandler1)
    filehandler1.close()
    '''
    
    # -- loading decision Tree model 
    
    file = open("DT_gs.model",'rb')
    DT_gs = pickle.load(file)
    file.close()
    
    print('Best Decition Tree parameters:',DT_gs.best_params_,'best score:',f'{DT_gs.best_score_:0.4f}')
    
    # --- Random Forest with Grid Search
    '''
    forest_params = {
        'n_estimators': [100, 300, 600]
    }
    
    clf_rf = GridSearchCV(RandomForestClassifier(),param_grid = forest_params , verbose=1)
    clf_rf.fit(X_train, y_train)
    
    filehandler3 = open("RandomForest_gs.model","wb")
    pickle.dump(clf_rf,filehandler3)
    filehandler3.close()
    
    '''
    # -- loading Random Forest model 
    
    
    # RandomForest_gs.model is in a rar , its around 200mb so it needs to be unzipped
    file2 = open("RandomForest_gs.model",'rb')
    rf_gs = pickle.load(file2)
    file2.close()
    
    
    print('------------------------------------')
    print('best score for RandomForest :',f'{rf_gs.best_score_:0.4f}','best params:',rf_gs.best_params_)
    
    
    #----- Neural Network - mlp
    '''
    nn_params ={
        'hidden_layer_sizes':[(100,),(96,),(96,96),(96,96,96)]
    }
    
    clf_NN = GridSearchCV(MLPClassifier(random_state=1,max_iter=400),param_grid = nn_params , verbose=1)
    clf_NN.fit(X_train, y_train)
    
    filehandler4 = open("NeuralNetwrok_GsCv.model","wb")
    pickle.dump(clf_NN,filehandler4)
    filehandler4.close()
    
    '''
    
    #-- Loading Neural Network model
    file3 = open("NeuralNetwrok_GsCv.model",'rb')
    clf_NN = pickle.load(file3)
    file3.close()
    print('------------------------------------')
    print('Neural Network best score:',f'{clf_NN.best_score_:0.4f}','best params:',clf_NN.best_params_)
    
    print('------------------------------------','\n\n')





