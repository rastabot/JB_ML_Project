'''
loads the  csv and applys train-test split, 
returns a tupple (train data, test data)
'''
import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_data():
    print('in train test data split......')
    df = pd.read_csv('adult.data')
    X = df.drop(' <=50K',axis=1)
    y = df[' <=50K']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    
   
    
    return df_train,df_test

