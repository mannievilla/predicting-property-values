############################ IMPORTS #########################


#standard ds imports
import pandas as pd
import numpy as np
import os

#visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

#import custom modules
from env import get_db_url

#import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

#ignore warnings
import warnings
warnings.filterwarnings("ignore")


def df_shape():
    
    print(f'The number of rows is {len(acquire_zillow())}')  
    print(f'The number of columns is {len(acquire_zillow().columns.to_list())}')

############################## AQUIRE ZILLOW FUNCTION ##############################


def acquire_zillow():
    '''
    This function checks to see if zillow.csv already exists, 
    if it does not, one is created
    '''
    #check to see if zillow.csv already exist
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    
    else:
        
        url = get_db_url('zillow')
        df = pd.read_sql('''SELECT *
                            FROM properties_2017 AS prop
                            JOIN predictions_2017 AS pred ON prop.parcelid = pred.parcelid
                            WHERE prop.propertylandusetypeid = 261;''', url)
        #creates new csv if one does not already exist
        df.to_csv('zillow.csv')

    
    return df
    




############################ PREPARE ZILLOW FUNCTION ###########################

def prep_zillow(df):
    '''
    This function takes in the zillow df
    then the data is cleaned and returned
    '''
    
    
    #create a list of columns to drop
    columns_to_drop = ['id', 'parcelid', 'airconditioningtypeid', 'architecturalstyletypeid',
       'basementsqft', 'buildingclasstypeid',
       'buildingqualitytypeid', 'calculatedbathnbr', 'decktypeid',
       'finishedfloor1squarefeet', 
       'finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15',
       'finishedsquarefeet50', 'finishedsquarefeet6', 'fireplacecnt',
       'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'hashottuborspa',
       'heatingorsystemtypeid', 'latitude', 'longitude', 'lotsizesquarefeet',
       'poolcnt', 'poolsizesum', 'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
       'propertycountylandusecode', 'propertylandusetypeid',
       'propertyzoningdesc', 'rawcensustractandblock', 'regionidcity',
       'regionidcounty', 'regionidneighborhood', 'regionidzip', 'roomcnt',
       'storytypeid', 'threequarterbathnbr', 'typeconstructiontypeid',
       'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
       'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
       'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
       'censustractandblock', 'id.1', 'parcelid.1', 'logerror',
       'transactiondate']

    
    # using the created list to drop columns
    df = df.drop(columns = columns_to_drop)
   


#     Define the mapping of old values to new values
#     value_mapping = {6037.0: 'Los Angeles', 6059.0: 'Orange', 6111.0: 'Ventura'}
    
    
    
#     # adding a county column using the mapping for counties
#     df['county'] = df['fips'].map(value_mapping) 
    
    
    
    
    # creating empty list for loop
    zip_list = []
    
    
    # bringing un touched DataFrame
    df_raw = acquire_zillow()
    
    for i in range(len(df_raw.regionidzip)):
        if not pd.isnull(df_raw.regionidzip[i]):
            zip = round(df_raw.regionidzip[i])
            zip_list.append(zip)
        else: 
            zip_list.append(np.nan)
            
    # remove homes with these bedroom and bathroom counts    
     
        
        
    df['zip_code'] = zip_list
    
    #drop null values
    df.dropna(subset=['zip_code'], inplace=True)

    
    #drop duplicates
    df.drop_duplicates(inplace=True)
    
    #let's rename the columns to be more readable
    df = df.rename(columns = {'bedroomcnt':'bedrooms', 
                              'bathroomcnt':'bathrooms', 
                              'calculatedfinishedsquarefeet':'sqft',
                              'taxvaluedollarcnt':'home_value'
                              })
    
    df = df.dropna()
    
    df = df[(df['bedrooms']<7) & (df['bathrooms']<8)] 
    
    
    return df


############################ WRANGLE ZILLOW FUNCTION ############################

def wrangle_zillow():
    '''
    This function acquires and prepares our Zillow data
    and returns the clean dataframe
    '''
    
    df = prep_zillow(acquire_zillow())
    return df



############################ SPLIT ZILLOW FUNCTION ############################

def split_zillow(df):
    '''
    This function takes in the dataframe
    and splits it into train, validate, test datasets
    '''    
    # train/validate/test split
    train_validate, test = train_test_split(df, test_size=.2, random_state=13)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=13)
    
    return train, validate, test


########################################## AQUIRE ZILLOW FUNCTION #######################################

def split_clean_zillow():
    '''
    This function splits our clean dataset into 
    train, validate, test datasets
    '''
    train, validate, test = split_zillow(wrangle_zillow())
    
#     print(f"train: {train.shape}")
#     print(f"validate: {validate.shape}")
#     print(f"test: {test.shape}")
    
    return train, validate, test




########################################## X Y SPLIT ##########################################


#create a function to isolate the target variable
def X_y_split():
    '''
    This function takes in a dataframe and a target variable
    Then it returns the X_train, y_train, X_validate, y_validate, X_test, y_test
    and a print statement with the shape of the new dataframes
    '''  
    train, validate, test = split_zillow(wrangle_zillow())

    X_train = train.drop(columns= 'home_value')
    y_train = train['home_value']

    X_validate = validate.drop(columns= 'home_value')
    y_validate = validate['home_value']

    X_test = test.drop(columns= 'home_value')
    y_test = test['home_value']
        
    # Have function print datasets shape
    # print(f'''
    # X_train -> {X_train.shape}
    # X_validate -> {X_validate.shape}
    # X_test -> {X_test.shape}''')
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test
        