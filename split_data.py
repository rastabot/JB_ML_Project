'''
loads the  csv and applys train-test split, 
returns a tupple (train data, test data ,data for graphs)
in RAW csv format
'''
import pandas as pd

from sklearn.model_selection import train_test_split

from clean_data import get_clean_df

def train_test_data():
    #print('in train test data split......')
    
    df_raw = pd.read_csv('adult.data')
    
    df = get_clean_df(df_raw)[0]
    
    #df_graphs = get_clean_df(df_raw)[1]
    
    
    X = df.drop('less than 50K',axis=1)
    y = df['less than 50K']
    
    #X_graphs = df_graphs.drop('less than 50K',axis=1)
    #y_graphs = df_graphs['less than 50K']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    
    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    #df_graphs2 = pd.concat([X_graphs,y_graphs],axis=1)
   
    
    return df_train,df_test,#df_graphs2

