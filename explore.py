# imports
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import wrangle


################################# Bed and Bath Combos over 1 Million #################################

def combo_bed_bath_1M():
    # splits data
    train, validate, test = wrangle.split_clean_zillow()

    # creates dataframe for sorted by max average home values
    mean_values = pd.DataFrame(train.groupby(['bedrooms', 'bathrooms'])['tax_value'].mean()).sort_values('tax_value', ascending=False)#.sort_values('tax_value', ascending=False))


    # Resetting the indices
    mean_values_over_1M = mean_values[mean_values['tax_value']>1e6]

    # resetting the index
    df_reset = mean_values_over_1M.reset_index()

    # Combining the indices into a single column
    df_reset['Combined'] = mean_values_over_1M.index.to_list()


    plt.figure(figsize=(16, 4))
    sns.barplot(x=df_reset.Combined, y='tax_value', data=df_reset)
    #plt.title(f'Values Ranging from \${round(df.tax_value.iloc[0])} to \${round(df.tax_value.iloc[len(df)-1])}')
    plt.xticks(rotation=90)
    plt.show()



################################# bedrooms over 1 Million #################################
    
    
def bed_bath_count_over_1M():
    
    # splits data
    train, validate, test = wrangle.split_clean_zillow()

    # creates dataframe for sorted by max average home values
    mean_values = pd.DataFrame(train.groupby(['bedrooms', 'bathrooms'])['tax_value'].mean()).sort_values('tax_value', ascending=False)#.sort_values('tax_value', ascending=False))


    # Resetting the indices
    mean_values_over_1M = mean_values[mean_values['tax_value']>1e6]

    # create list to hold bedrooms and bath
    x_list = []
    y_list = []
    # loop for the bed and bath list
    for x,y in mean_values_over_1M.index.to_list():
        x_list.append(x)
        y_list.append(y)


    x_list = pd.DataFrame(x_list)
    x_list = pd.DataFrame(x_list.value_counts(), columns=['count'])

    y_list = pd.DataFrame(y_list)
    y_list = pd.DataFrame(y_list.value_counts(), columns=['count'])

    return x_list, y_list



################################# value and sqft #################################



def relplot_var(df):
    
    plt.figure(figsize=(20, 9))
    sns.set(style="ticks")
    sns.relplot(df, x='sqft', y='tax_value', kind='scatter')
    plt.show()



################################# bedrooms 1 and 2 #################################


def one_two_bedrooms():


    # splits data
    train, validate, test = wrangle.split_clean_zillow()


    # creates list 
    bed_2_train = train[train['bedrooms']==1]
    bed_1_train = train[train['bedrooms']==2]

    # Create subplots with 1 row and 2 columns and assign axes to variables
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot DataFrame 1 in ax1
    sns.scatterplot(data=bed_2_train, x='sqft', y='tax_value', ax=ax1)
    ax1.set_title('Two Bedrooms')

    meanline2 = bed_2_train['tax_value'].mean()
    ax1.axhline(meanline2, color='red', linestyle='--', label='Mean')

    # Plot DataFrame 2 in ax2
    sns.scatterplot(data=bed_1_train, x='sqft', y='tax_value', ax=ax2)
    ax2.set_title('One Bedroom')

    meanline1 = bed_1_train['tax_value'].mean()
    ax2.axhline(meanline1, color='red', linestyle='--', label='Mean')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()
