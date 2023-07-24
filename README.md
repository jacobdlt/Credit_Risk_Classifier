# Credit Risk Classification

## Overview

The purpose of this analysis is to build a supervised machine learning model that can identify the creditworthiness of borrowers.

Using lending data that included information about borrowers such as loan size, income, number of accounts, interest rate, etc. I predicted whether they fell into a "healthy loan" or "high risk" category.

1.) I first read the lending data into a Pandas dataframe from the .csv file and created features (X) and labels (y) sets from the columns, with the loan_status column being the label and the rest features. 

2.) Secondly, I used the .value_counts function to determine the balance of the label variable. Healthy loans (0) heavily outnumbered high-risk loans (1):

![image of .value_counts](/code/value_counts.jpg)

3.) I then split the data into training and testing datasets using the train_test_split function from scikit-learn in order to fit a Logistic Regression model with the testing data.

4.) After generating predictions for the testing data labels (y_test) with this model, I saved the results (y_pred) and generated a confusion matrix and classification report to evaluate it's performance.

Lastly, I repeated this process from step 2 after using RandomOverSampler to oversample the high-risk loan label (1) to equal the number of healthy loans (0) to see if this would enhance the model's predictive power.


## Results

* Model 1:

![image of model 1 results](/code/model1_results.jpg)


* Model 2:

![image of model 2 results](/code/model2_results.jpg)

## Summary

Both models had perfect precision for healthly loans, however the oversampled model (Model 2) was slightly less precise at predicting high-risk loans (84% vs Model 1's 85%).

Model 2 was more accurate with a 99% accuracy score compared to Model 1's 95%.

Recall for model 1 was also worse for high-risk loans (91%) vs Model 2 (99%).

In this case, being able to predict high-risk loans is more important than predicting healthy ones. Failing to categorize loans as high-risk could lead to unpreparedness in the case of defaults and therefore potential liquidity / solvency risks for the business. This could also lead to lending to unqualified borrowers

With this in mind, I would recommend the use of Model 2 exclusively since accuracy and recall were better and precision essentially the same as Model 1. 
