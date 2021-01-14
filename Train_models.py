'''
 cross-validation  and traing 3 models:
    dessision trees, Random Forest and Neural Networks
'''

import numpy as np
import pandas as pd
import seaborn as sb
import sklearn.linear_model as skl
import matplotlib.pyplot as plt

import pickle

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split,KFold,cross_val_score,StratifiedKFold,GridSearchCV
                                    
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
                            
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier

from sklearn.neural_network import MLPClassifier    

from Clean_Data import get_clean_df

df = get_clean_df()

# --- Train, Test split
X = df.drop('less than 50K',axis=1)
y = df['less than 50K']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

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

# -- loading the model 

file = open("DT_gs.model",'rb')
DT_gs = pickle.load(file)
file.close()

print('Best D.T parameters:',DT_gs.best_params_,'best score:',DT_gs.best_score_)

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
# -- loading the model 

'''
# RandomForest_gs.model is in a rar , its around 200mb so it needs to be unzipped
file2 = open("RandomForest_gs.model",'rb')
rf_gs = pickle.load(file2)
file2.close()


print('------------------------------------')
print('best score for RandomForest :',rf_gs.best_score_,'best params:',rf_gs.best_params_)
'''

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
file3 = open("NeuralNetwrok_GsCv.model",'rb')
clf_NN = pickle.load(file3)
file3.close()
print('------------------------------------')
print('Neural Network best score:',clf_NN.best_score_,'best params:',clf_NN.best_params_)





