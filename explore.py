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
    mean_values = pd.DataFrame(train.groupby(['bedrooms', 'bathrooms'])['home_value'].mean()).sort_values('home_value', ascending=False)#.sort_values('tax_value', ascending=False))


    # Resetting the indices
    mean_values_over_1M = mean_values[mean_values['home_value']>1e6]

    # resetting the index
    df_reset = mean_values_over_1M.reset_index()

    # Combining the indices into a single column
    df_reset['Combined'] = mean_values_over_1M.index.to_list()
    


    plt.figure(figsize=(16, 4))
    sns.barplot(x=df_reset.Combined, y='home_value', data=df_reset)
    #plt.title(f'Values Ranging from \${round(df.tax_value.iloc[0])} to \${round(df.tax_value.iloc[len(df)-1])}')
    plt.xticks(rotation=90)
    plt.show()



################################# bedrooms over 1 Million #################################
    
    
def bed_bath_count_over_1M():
    
    # splits data
    train, validate, test = wrangle.split_clean_zillow()

    # creates dataframe for sorted by max average home values
    mean_values = pd.DataFrame(train.groupby(['bedrooms', 'bathrooms'])['home_value'].mean()).sort_values('home_value', ascending=False)#.sort_values('tax_value', ascending=False))


  
    mean_values_over_1M = mean_values[mean_values['home_value']>1e6]

    # create list to hold bedrooms and bath
    x_list = []
    y_list = []
    # loop for the bed and bath list
    for x,y in mean_values_over_1M.index.to_list():
        x_list.append(x)
        y_list.append(y)

    # changing them to dataframes and addung the property counts
    x_list = pd.DataFrame(x_list)
    x_list = pd.DataFrame(x_list.value_counts(), columns=['Property Count'])

    y_list = pd.DataFrame(y_list)
    y_list = pd.DataFrame(y_list.value_counts(), columns=['Property Count'])
    # 
    
    y_list = y_list.reset_index()
    x_list = x_list.reset_index()
    # Set the modified column name with line break and center alignment
    x_list = x_list.rename(columns={0: 'Bedrooms'})
    y_list = y_list.rename(columns={0: 'Bathrooms'})
    
    
    x_list['Bedrooms'] = x_list['Bedrooms'].astype(int)
 
    
    x_list.Bedrooms = x_list.Bedrooms.round(0)
    y_list.Bathrooms = y_list.Bathrooms.round(1)
    
    # this step has to be preformed last because it changes to .styler and it is no longer a dataframe
#     y_list = y_list.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
#                                             'selector': 'th',
#                                             'props': [('text-align', 'center')]
#                                             }])

#     x_list = x_list.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
#                                         'selector': 'th',
#                                         'props': [('text-align', 'center')]
#                                         }])
   

    return x_list, y_list

################################# y_list and dataframe #################################


def y_and_df():


    x_list, y_list = bed_bath_count_over_1M()
 

    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Display the DataFrame on the left subplot
    axes[0].axis('off')  # Remove axis
    axes[0].table(cellText=y_list.values, colLabels=y_list.columns, loc='center')

    # Plot a bar plot on the right subplot
    sns.barplot(data=y_list, x='Bathrooms', y='Property Count', order=y_list['Bathrooms'], ax=axes[1])

    # Set labels and title for the bar plot
    axes[1].set_xlabel('Bathrooms')
    axes[1].set_ylabel('Property Count')
    axes[1].set_title('Property Count over 1M by Bathrooms')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()



################################# x_list and dataframe #################################



def x_and_df():


    x_list, y_list = bed_bath_count_over_1M()


    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Display the DataFrame on the left subplot
    axes[0].axis('off')  # Remove axis
    axes[0].table(cellText=x_list.values, colLabels=x_list.columns, loc='center')

    # Plot a bar plot on the right subplot
    sns.barplot(data=x_list, x='Bedrooms', y='Property Count', order=x_list['Bedrooms'], ax=axes[1])

    # Set labels and title for the bar plot
    axes[1].set_xlabel('Bedrooms')
    axes[1].set_ylabel('Property Count')
    axes[1].set_title('Property Count over 1M by Bedrooms')

    # Adjust the layout and spacing
    plt.tight_layout()

    # Show the plot
    plt.show()





################################# value and sqft #################################



def relplot_var(df):
    
    plt.figure(figsize=(20, 9))
    sns.set(style="ticks")
    sns.relplot(df, x='sqft', y='home_value', kind='scatter')
    plt.show()



################################# bedrooms 1 and 2 #################################


def one_bath_two_bedrooms():


    # splits data
    train, validate, test = wrangle.split_clean_zillow()


    # creates list 
    bath_1_train = train[train['bathrooms']==1]
    bed_2_train = train[train['bedrooms']==2]

    # Create subplots with 1 row and 2 columns and assign axes to variables
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot DataFrame 1 in ax1
    sns.scatterplot(data=bed_2_train, x='sqft', y='home_value', ax=ax1)
    ax1.set_title('Two Bedrooms')

    meanline2 = bed_2_train['home_value'].mean()
    ax1.axhline(meanline2, color='red', linestyle='--', label='Mean')

    # Plot DataFrame 2 in ax2
    sns.scatterplot(data=bath_1_train, x='sqft', y='home_value', ax=ax2)
    ax2.set_title('One Bathroom')

    meanline1 = bath_1_train['home_value'].mean()
    ax2.axhline(meanline1, color='red', linestyle='--', label='Mean')

    # Adjust the layout
    plt.tight_layout()

    # Show the plot
    plt.show()
    
    return meanline1, meanline2


def sf_div_value():
    
        # splits data
    train, validate, test = wrangle.split_clean_zillow()
    
        # creates list 
    bath_1_train = train[train['bathrooms']==1]
    bed_2_train = train[train['bedrooms']==2]
    
    bed_per_sf = bed_2_train['home_value'].sum() / bed_2_train['sqft'].sum()
    bath_per_sf = bath_1_train['home_value'].sum() / bath_1_train['sqft'].sum()
    
    return bath_per_sf, bed_per_sf
