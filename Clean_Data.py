'''
Pre-prosseising the data, feature analyzing, feature cleaning
and reading it for a data frame .
'''

#--------------
import numpy as np
import pandas as pd
import seaborn as sb

from split_data import train_test_data


#================
'''
this function returns a boolean for the column '<=50K'
converts:  <=50K  to  True  , >50K to False

'''
def fifty(val):
    return val == ' <=50K'

#+++++++++++++++++++
    


#-- set seaborn style
sb.set_style("darkgrid")
sb.set(rc={'figure.figsize':(11.7,8.27)})



def get_clean_df(df):
    
    '''
    returns a tuple , (df after get_dummies, df before get_dummies)
    '''

    #df = pd.read_csv('adult.data')
    
    # -------- renaming columns
    df.rename(columns={
        ' 40':'hours_per_week',
        ' United-States':'Country',
        'Not-in-family':'Relationship',
        '39':'Age',
        ' 77516':'fnlwgt',
        ' 13':'education_num',
        ' 2174':'capital_gain',
        ' 0':'capital_loss',
        ' White':'Ethnicity',
        ' Male':'Gender',
        '':'Occupation',
        ' Not-in-family':'Relationship',
        ' Never-married':'Marital',
        ' State-gov':'Workclass',
        ' Adm-clerical':'Occupation',
        ' Bachelors':'Bachelors'
        },inplace=True)
    
    #-------renaming and reasiging TARGET( 'y'  column)
    
    df['less than 50K'] = df[[' <=50K']].apply(fifty,axis=1)
    
    #----- True to 1 , False to 0---
    df['less than 50K'] = df['less than 50K'].astype(int)
    
    #---cleaning features
    
    df['Workclass'].replace(' ?',' Private',inplace=True)
    df['Workclass'].replace(' Private','Private',inplace=True)
    df['Occupation'].replace(' ?',' Other-service',inplace=True)
    
    #-- finalizing the DataFrame
    
    df.drop(['fnlwgt','education_num',' <=50K'],axis=1,inplace=True)
    #print(df.columns)
    
    
    
    clean_df = pd.get_dummies(df,columns=['Relationship',
                                          'Workclass',
                                          'Bachelors',
                                          'Marital',
                                          'Occupation',
                                          'Ethnicity',
                                          'Gender',
                                          'Country'],drop_first=True)
    
    
    print('Data Frame is clean and ready to synthethize')
    print('*******************************************')
    
    return clean_df, df











