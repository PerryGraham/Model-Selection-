# - - Work in progress - - 
# Model Selection For Beginners
---------------------------------------------------
> ###### By: Graham Pinsent 

Purpose of this notebook: 
* Summary of machine learning models (from scikit-learn)
* Focuses on the applications of the models
* Reading through documentation about these models can often be very intimidating for beginners, especially when you do not have a strong math background. This is meant to be written for more people to understand the difference between models 
* I am not an expert by any means, some say one of the best ways to learn is by teaching. 

#### Table of Contents 

* [Regression](#Regression)
  * Linear
   * [LinearRegression](#LinearRegression)
   * [RidgeRegression](#RidgeRegression)
   * [Lasso](#Lasso)
   * [Elastic-Net](#Elastic-Net)
   * [Bayesian](#Bayesian)
* [Classification](#Classification)

### Regression 
Regression problems invole predicting a value that is continuous. For example predicting the weight of something, or monthly sales. There is a range of possible outcomes. 

##### [LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

`from sklearn.linear_model import LinearRegression`

Also known as: Ordinary Least Squares. Line of best fit where the average distance to each data point is the lowest possible with a straight line. 

* Low bias 
* High Varriance 
* Highly sensitive to random y_train errors (outliers in the target data)
* Assumes no correlation between features, which is rare.

This is a simple, quick to run, model that will not pick up on complex relationships between features and the target. However, when there is not a lot of data or there is large noise in the data, this can often produce better results than more complex models. 

[Prarameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html):
* fit_intercept = (defualt-True) if false, the y-intercept=0
* normalize = (default-FALSE) if true, X will be subtracted by the mean and divided by l2-norm. 
* copy_X = (default-True) if true, X is copied not overwritten
-------------


##### [RidgeRegression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

`from sklean.linear_model import RidgeCV`

This is the similar to linear regression but has a way to reduce overfitting to training data. Instead of getting the lowest mean to each datapoint, its offsets slightly (alpha value) based on testing results from cross validation. Instead of optimizing for best fit to training data, it's optimizing for lowest cross validation scores. This is known as L2 regularisation. This model shrinks all of the correlated features.  

* Good to use when the data has multicollinearity (when there is correlation between features 
* Relationship is linear, little outliers, and independence

The Ridge model on sklearn has build in cross validation to find the best alpha value.

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV):
* alphas = (default 0.1, 1.0, 10.0) ndarray of alpha values, larger values mean stronger regularization 
* scoring = (default NONE) [Here](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules) for all the types of scoring methods
* cv = (default NONE) cross validation method. int is number of folds

[Read more](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf)

---------

##### [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

`from sklearn.linear_model import Lasso`

Very similar to RidgeRegression, however Lasso is able to reduce (all the way to 0) the infulence of features that dont provide any valuable insight for the target. However. is not as good when all the variables are usefull. This model picks one of the correlated features and removes the others. This is known as L1 regularisation. 

* Can be used to help feature selection 
* Works well when there are a lot of useless features 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso):
* alpha = (default 1) refer to ridge
* random_state = (default none) int for seeding the random number generator

Also see [Multi-task Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso)

--------



##### [Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

`from sklearn.linear_model import ElasticNet`

Is a combination of both RidgeRegression and Lasso, combining both L1 and L2 regularisation. It preforms cross validation to find the best mix of both RidgeRegression and Lasso. This model is able to both shrink features aswell as remove them from the equation

* Strong when handling correlation between features 
* Good If you are unsure if you need L1 or L2 regularisation

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)
* alpha = (default 1) refer to ridge 
* l1_ratio = (default 0.5) where 0 is more l2 favoured and 1 is l1 favoured 

Also see [Multi-task Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)

--------

##### [Bayesian](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)

`from sklean.linear_model import BayesianRidge`

Baysian allows you to represent uncertainty. This approach is able to create trends based off old data and update its trends based on new data, assuming that the model is not perfect and needs to adjust for error. As the amount of data increase toward infinite the model gets closer to LinearRegression. 

* Good when the target values will not fall within the same x values as your training data. 
* For example in time series when you are predicting future targets
* Or perhaps for sports metrics models predicting games played after a long break (COVID) where there is expected to be uncertainty

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge):
* Using the correct weights for the lambda and alpha values require optimization and regularization. Refer [Here](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression) for a user guide 


------

### Classification
Problems where you are trying to predict if an outcome will happen or not (a true of false outcome). For example, if a team will win or lose a match. 

References: 
   * [Scikit-learn](https://scikit-learn.org/stable/index.html)
   * [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
   * Search results, stackoverflow, and wikipedia
