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

from explore import combo_bed_bath_1M, bed_bath_count_over_1M, relplot_var, one_bath_two_bedrooms, x_and_df, y_and_df
from explore import sf_div_value


from wrangle import X_y_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, TweedieRegressor, LassoLars
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures




def scale_data():
    
    # x and y split
    X_train, y_train, X_validate, y_validate, X_test, y_test = X_y_split()
    
    to_scale = X_train.columns.tolist()
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled


def median_mean_choice():
    
    # x and y split
    X_train, y_train, X_validate, y_validate, X_test, y_test = X_y_split()
    
    # make y_train a DataFrame to abel to attach a new column
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    # 
    
    # adding the baseline column to dataframe usng the mean and meadian of infant moratility as features
    y_train['baseline_mean'] = y_train['home_value'].mean()
    y_train['baseline_median'] = y_train['home_value'].median()
    
    y_validate['baseline_mean'] = y_train['home_value'].mean()
    y_validate['baseline_median'] = y_train['home_value'].median()
    
    # scores
    rmse_mean = mean_squared_error(y_train['home_value'],
                                   y_train['baseline_mean'], squared=False) ** .5
    rmse_med = mean_squared_error(y_train['home_value'],
                                   y_train['baseline_median'], squared=False) ** .5


    if rmse_mean < rmse_med:
        
        
        rmse_train_mean = mean_squared_error(y_train['home_value'], y_train.baseline_mean) ** .5

        rmse_validate_mean = mean_squared_error(y_validate['home_value'], y_validate.baseline_mean) ** .5
        
        
        # Let's house our metrics in a df to later compare
        metric_df = pd.DataFrame(data = [
                {
                    "model": "mean_baseline",
                    "RMSE_train": rmse_train_mean,
                    "RMSE_validate":rmse_validate_mean,
                    "R2_validate": explained_variance_score(y_validate['home_value'], y_validate.baseline_mean)
                }
        ])
        
        
        # MAKE THE THING: create the model object
        lm = LinearRegression()

        #1. FIT THE THING: fit the model to training data
        OLSmodel = lm.fit(X_train, y_train['home_value'])

        #2. USE THE THING: make a prediction
        y_train['value_pred_lm'] = lm.predict(X_train)

        #3. Evaluate: RMSE
        rmse_train = mean_squared_error(y_train['home_value'], y_train.value_pred_lm) ** .5
        
        # predict validate
        y_validate['value_pred_lm'] = lm.predict(X_validate)

        # evaluate: RMSE
        rmse_validate = mean_squared_error(y_validate['home_value'], y_validate.value_pred_lm) ** .5
        
        
        #Append this to the metric_df

        metric_df = metric_df.append({
            "model":"OLS Regressor",
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "R2_validate": explained_variance_score(y_validate['home_value'], y_validate.value_pred_lm)
        }, ignore_index=True)
        
        

        # MAKE THE THING: create the model object
        lars = LassoLars(alpha=.1) # you can loop through for higher values to chane som hyperparameters

        #1. FIT THE THING: fit the model to training data
        # We must specify the column in y_train, since we have converted it to a dataframe from a series!
        lars.fit(X_train, y_train['home_value'])

        #2. USE THE THING: make a prediction
        y_train['value_pred_lars'] = lars.predict(X_train)

        #3. Evaluate: RMSE
        rmse_train = mean_squared_error(y_train['home_value'], y_train.value_pred_lars) ** (1/2)

        #4. REPEAT STEPS 2-3

        # predict validate
        y_validate['value_pred_lars'] = lars.predict(X_validate)

        # evaluate: RMSE
        rmse_validate = mean_squared_error(y_validate['home_value'], y_validate.value_pred_lars) ** (1/2)      
        
        
        #Append this to the metric_df

        metric_df = metric_df.append({
            "model":"LassoLars",
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "R2_validate": explained_variance_score(y_validate['home_value'], y_validate.value_pred_lars)
        }, ignore_index=True)
        
        
        # MAKE THE THING: create the model object
        glm = TweedieRegressor(power=.5, alpha=0) # loop through to change the power and alpha

        #1. FIT THE THING: fit the model to training data
        # We must specify the column in y_train, since we have converted it to a dataframe from a series!
        glm.fit(X_train, y_train['home_value'])

        #2. USE THE THING: make a prediction
        y_train['value_pred_glm'] = glm.predict(X_train)

        #3. Evaluate: RMSE
        rmse_train = mean_squared_error(y_train['home_value'], y_train.value_pred_glm) ** .5

        #4. REPEAT STEPS 2-3

        # predict validate
        y_validate['value_pred_glm'] = glm.predict(X_validate)

        # evaluate: RMSE
        rmse_validate = mean_squared_error(y_validate['home_value'], y_validate.value_pred_glm) ** .5
        
        #Append this to the metric_df

        metric_df = metric_df.append({
            "model":"Tweedie",
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "R2_validate": explained_variance_score(y_validate['home_value'], y_validate.value_pred_glm)
        }, ignore_index=True)
        
        
        #1. Create the polynomial features to get a new set of features
        pf = PolynomialFeatures(degree=2) #Quadratic aka x-squared

        #1. Fit and transform X_train_scaled
        X_train_degree2 = pf.fit_transform(X_train)

        #1. Transform X_validate_scaled & X_test_scaled 
        X_validate_degree2 = pf.transform(X_validate)
        X_test_degree2 = pf.transform(X_test)

        
        #2.1 MAKE THE THING: create the model object
        lm2 = LinearRegression()

        #2.2 FIT THE THING: fit the model to our training data. We must specify the column in y_train, 
        # since we have converted it to a dataframe from a series! 
        lm2.fit(X_train_degree2, y_train['home_value'])

        #3. USE THE THING: predict train
        y_train['value_pred_lm2'] = lm2.predict(X_train_degree2)

        #4. Evaluate: rmse
        rmse_train = mean_squared_error(y_train['home_value'], y_train.value_pred_lm2) ** .5

        #5. REPEAT STEPS 3-4

        # predict validate
        y_validate['value_pred_lm2'] = lm2.predict(X_validate_degree2)

        # evaluate: rmse
        rmse_validate = mean_squared_error(y_validate['home_value'], y_validate.value_pred_lm2) ** .5
        
        
        #Append
        metric_df = metric_df.append({
            "model":"Polynomial",
            "RMSE_train": rmse_train,
            "RMSE_validate": rmse_validate,
            "R2_validate": explained_variance_score(y_validate['home_value'], y_validate.value_pred_lm2)
        }, ignore_index=True)

        
        
        plt.figure(figsize=(16,8))
        #actual vs mean
        plt.plot(y_validate['home_value'], y_validate.baseline_mean, alpha=.5, color="gray", label='_nolegend_')
        plt.annotate("Baseline: Predict Using Mean", (6e6, 15))

        #actual vs. actual
        plt.plot(y_validate['home_value'], y_validate['home_value'], alpha=.5, color="blue", label='_nolegend_')
        plt.annotate("The Ideal Line: Predicted = Actual", (8e6, 8.1e6), rotation=25.5)

        #actual vs. LinearReg model
        plt.scatter(y_validate['home_value'], y_validate.value_pred_lm, 
                    alpha=.5, color="red", s=100, label="Model: LinearRegression")
        #actual vs. LassoLars model
#         plt.scatter(y_validate['home_value'], y_validate.value_pred_lars, 
#                     alpha=.5, color="purple", s=100, label="Model: Lasso Lars")
        #actual vs. Tweedie/GenLinModel
#         plt.scatter(y_validate['home_value'], y_validate.value_pred_glm, 
#                     alpha=.5, color="yellow", s=100, label="Model: TweedieRegressor")
        #actual vs. PolynomReg/Quadratic
        plt.scatter(y_validate['home_value'], y_validate.value_pred_lm2, 
                    alpha=.5, color="green", s=100, label="Model 2nd degree Polynomial")
        plt.legend()
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Where are predictions more extreme? More modest?")
        plt.show()
        
        return metric_df

    else:
        
        print('Median Wins')
        
        
        
def test_run():
    
    X_train, y_train, X_validate, y_validate, X_test, y_test = X_y_split()
    
    y_train = pd.DataFrame(y_train)
    
    lm = LinearRegression()
    
    lm.fit(X_train, y_train)
    
    # Convert y_test Series to a df
    y_test = pd.DataFrame(y_test)

    # USE THE THING: predict on test
    y_test['value_pred_ols'] = lm.predict(X_test)

    # Evaluate: rmse
    rmse_test = mean_squared_error(y_test.home_value, y_test.value_pred_ols) ** (.5)

    print(f"""RMSE for LassoLars alpha=0.01
    _____________________________________________      
    Out-of-Sample Performance: {rmse_test}
    Baseline: {y_train.home_value.mean()}""")
