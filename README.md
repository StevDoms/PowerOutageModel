# Power Outage Model Analysis

This project was done as part of UCSD's DSC80 class.

## Framing the Problem

The goal of our analysis is to predict the **cause category** of an outage based on the **outage duration**, **numbers of customers affected**, **total price** of electricity during the outage, **NERC region**, and **climate category**. Since **cause category** is a categorical variable with 7 unique values, we are going to be performing a multiclass classification. We chose to predict the cause category because it allows for different strategies and interventions to be implemented in hopes to mitigate future outages. If severe weather is a prevalent cause, reinforcing infrastructure against extreme conditions might be a priority task to complete. We chose to use F_1 score with weighted average because it considers both the precision and recall, and also accounts for the cause imbalance. Since mitigation plans costs a lot of monetary value, we want to balance out the false positive and false negative to ensure that the mitigation targets the right causes. We don't use other metrics such as accuracy because the causes categories are unevenly distributed which would not provide a comprehensive understanding of the model. 

## Baseline Model

For our model, we used a decision tree classifier. We only specified the criterion to be 'entropy' as our hyperparameters, and as a result, our model uses splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None. Our model takes in 3 quantitative variables which are outage duration, numbers of customers affected and total price and 2 nominal variables which are NERC region and climate category as our regressors to predict cause category which is a nominal variable. We performed one hot encoding for the NERC region and climate category by using the OneHotEncoder class from sklearn and kept all the quantitative categories as is. </br> </br>

