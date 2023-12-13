# Power Outage Model Analysis

This project was done as part of UCSD's DSC80 class.

Made by: Kenneth Hidayat, Steven Dominic Sahar

> Our exploratory data analysis on this dataset can be found [here](https://stevdoms.github.io/PowerOutageImpactAnalysis/).

## Framing the Problem

The goal of our analysis is to predict the **cause category** of an outage based on the **outage duration**, **numbers of customers affected**, **total price** of electricity during the outage, **NERC region**, and **climate category**. Since **cause category** is a categorical variable with 7 unique values, we are going to be performing a multiclass classification. We chose to predict the cause category because it allows for different strategies and interventions to be implemented in hopes to mitigate future outages. If severe weather is a prevalent cause, reinforcing infrastructure against extreme conditions might be a priority task to complete. We chose to use F_1 score with weighted average because it considers both the precision and recall, and also accounts for the cause imbalance. Since mitigation plans costs a lot of monetary value, we want to balance out the false positive and false negative to ensure that the mitigation targets the right causes. We don't use other metrics such as accuracy because the causes categories are unevenly distributed which would not provide a comprehensive understanding of the model. 

## Baseline Model

For our model, we used a decision tree classifier. We only specified the criterion to be 'entropy' as our hyperparameters, and as a result, our model uses splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None. Our model takes in 3 quantitative variables which are outage duration, numbers of customers affected and total price and 2 nominal variables which are NERC region and climate category as our regressors to predict cause category which is a nominal variable. We performed one hot encoding for the NERC region and climate category by using the OneHotEncoder class from sklearn and kept all the quantitative categories as is.

We evaluated the performance of our baseline model using f1_score, and our model received an f1_score of 0.6118824440755535. This suggests that the harmonic mean of precision and recall to be 0.6118, indicating a reasonably good balance between precision and recall. As such, our current model is pretty good.

## Final Model

We used the exact same features that was used for the baseline model. We transformed both the nominal variables with One Hot Encoding. However, instead of leaving the quantitative variables as is, we applied StandardScalar() to the Total Price and Outage Duration and QuantileTransformer() to the numbers of customers affected. We applied StandardScaler() to Total Price and Outage Duration to standardize the scale and ensure that features with different magnitudes don’t dominate and aids the model in converging faster. This transformation also helps in normalizing distributions and reducing the impact of outliers, resulting in improved stability. We applied QuantileTransformer() for numbers of customers affected to transform the skewed distributions so that the feature becomes less sensitive to extreme values and better suited for the model. By enhancing these features comparability, these transformations would improve our overall model performance.

To standardize the comparison with the baseline model, we used a decision tree classifier for our model. We used GridSearchCV with 8 cross-validations to select the max_depth and min_samples_split hyperparameters. Based on this process, the hyperparameter that ended up performing best is max_depth=8 & min_samples_split=50, with criterion='entropy' (which is the same as the baseline model).

We evaluated the performance of our final model using f1_score, and our model received an f1_score of 0.6626287617698571. Our final model produced a higher f1 score than our baseline model which indicates an improvement in the final model's ability to balance precision and recall compared to the baseline model. This further suggests that the final model is better at capturing true positives while minimizing false positives and false negatives which results in a more reliable and accurate prediction of the cause category of outages.

## Fairness Analysis

We conducted a permutation test for groups which are outages above and below 24 hours. In addition, we used f1_score as our evaluation metric and a significance level of 5%.

**Null hypothesis**: My model is fair. The f1_score for outages above 24hours (1440minutes) is the same as f1_score for outages below 24hours, and any differences are due to random chance

**Alternative hypothesis**: My model is unfair. The f1_score is different for outages above and below 24 hours

**Test Statistic**: Absolute difference between the f1_score of outages above and below 24 hours

Based on our permutation test, we obtain a p-value of 0.0% which is less than our set threshold of 5%. Hence, we reject our null hypothesis, and conclude that there is a statistical difference between the f1_score of outages above and below 24 hours.

