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