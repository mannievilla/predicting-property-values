# Predicting Home Value
 
# Project Description
 
Your home is almost everyone biggest invest they make in their life. In this project I will be explore Zillow data to predict property values.
 
# Project Goal
 
* Discover drivers that have a string relationship to predict values.
* Use drivers to develop a machine learning model to make accurate predictions. 
* Home values will be compared to actual home values. 
* This information could be used to better understand changing markets.
 
# Initial Thoughts
 
My initial hypothesis is that drivers of home vlaue will be location, square feet, and bedroom size.
 
# The Plan
 
* Aquire data from database
 
* Prepare data
   * Remove Nulls from data
       * ---
       * ---
       * ---
       * ---
       * ---
 
* Explore data in search of drivers to predict property value
   * Answer the following initial questions
       * What number of Bedrooms and Bathrooms do Homes averaging over 1 million have?
       * How does Square footage impact Home Value?
       * Is having 1 bathroom worse for property value than having 2 bedrooms?
       * What are properties Home Value averages in their counties ?
      
* Develop a Model to predict property values
   * Use drivers identified in explore to build predictive models of different types
   * Evaluate models on train and validate data
   * Select the best model based on highest accuracy
   * Evaluate the best model on test data
 
* Draw conclusions
 
# Data Dictionary


|**Feature**|**Description**|
|:-----------|:---------------|
|Bedrooms | Numbers of bedrooms|
|Bathrooms | Numbers of bathrooms|
|sqft | Total calculated square footage|
|fips | Cunty Codes|
|home_value | Assessed property value|
|zipcode | Zip Code|


# Steps to Reproduce
1) Clone this repo.
2) Acquire the data from CodeUp database.
3) Run notebook.
 
# Takeaways and Conclusions
* Square Footage plays big role towards value.
* Number of Bedrooms plays a big role towards.
* Although bahtrooms does have a relation with value it is tied to Bedrooms.
* Million dollar homes still follow a standard. More doesn't mean more.

 
# Recommendations
* Removing and unnecsaary columns and adding a exploring other features like lot size or making a new like the difference of lot and home site square footage. Removing outliers like property values over 1 million dollars.

# Next Steps
* Run RFE or KBest for best feaures to use in the model and create new ones with. And ivestigate why Tweedie ran so poor.
