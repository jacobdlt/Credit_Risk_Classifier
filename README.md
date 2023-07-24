# Credit Risk Classification

## Overview

The purpose of this analysis is to build a supervised machine learning model that can identify the creditworthiness of borrowers.

Using lending data that included information about borrowers such as loan size, income, number of accounts, interest rate, etc. I predicted whether they fell into a "healthy loan" or "high risk" category.

1.) I first read the lending data into a Pandas dataframe from the .csv file and created features (X) and labels (y) sets from the columns, with the loan_status column being the label and the rest features. 

2.) Secondly, I used the .value_counts function to determine the balance of the label variable. Healthy loans (0)heavily outnumbered high risk-loans (1):

![image of .value_counts](/code/value_counts.jpg)

3.) I then split the data into training and testing datasets using the train_test_split function from scikit-learn in order to fit a Logistic Regression model with the testing data.

4.) After generating predictions for the testing data labels (y_test) with this model, I saved the results (y_pred) and generated a confusion matrix and classification report to evaluate it's performance.

Lastly, I repeated this process from step 2 after using RandomOverSampler to oversample the high-risk loan label (1) to equal the number of healthy loans (0) to see if this would enhance the model's predictive power.


## Results

* Machine Learning Model 1:

![image of model 1 results](/code/model1_results.jpg)


* Machine Learning Model 2:

![image of model 2 results](/code/model2_results.jpg)

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
