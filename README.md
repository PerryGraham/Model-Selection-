# Model Selection For Beginners
---------------------------------------------------
> ###### By: Graham Pinsent 

### - - Work in progress - - 
Things to add: 
- Nearest neighbors
- Decision trees 
- Ensemble methods?
    - random forest regression
    - random forest classifier 
    - bagging
* Fix ToC 

Purpose of this notebook: 
* Summary of machine learning models (from scikit-learn)
* Focuses on the applications of the models
* Reading through documentation about these models can often be very intimidating for beginners, especially when you do not have a strong math background. This is meant to be written for more people to understand the difference between models. 
* I am not an expert by any means, some say one of the best ways to learn is by teaching. Every model is a link to offical websites with more in depth information. 

#### Table of Contents 

* [Regression](#Regression)
    * [LinearRegression](#LinearRegression)
    * [RidgeRegression](#RidgeRegression)
    * [Lasso](#Lasso)
    * [Elastic-Net](#Elastic-Net)
    * [Bayesian](#Bayesian)
    * [Stochastic Gradient Decent (Regression)](#Stochastic-Gradient-Decent-(Regression))
    * [Support Vector Regression](#Support-Vector-Regression)
    
    
* [Classification](#Classification)
    * [LogisticRegression](#LogisticRegression)
    * [Stochastic Gradient Descent (Classifier)](#Stochastic-Gradient-Descent-(Classifier))
    * [Linear Discriminant Analysis](#Linear-Discriminant-Analysis)
    * [Quadratic Discriminant Analysis](#Quadratic-Discriminant-Analysis)
    * [Naive Bayes (Mulrinomial)](#Naive-Bayes-(Mulrinomial))
    * [Suppot Vector Classification](#Suppot-Vector-Classification)

### Regression 
Regression problems invole predicting a value that is continuous. For example predicting the weight of something, or monthly sales. There is a range of possible outcomes. 

#### [LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)

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


#### [RidgeRegression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)

`from sklean.linear_model import RidgeCV`

This is the similar to linear regression but has a way to reduce overfitting to training data. Instead of getting the lowest mean to each datapoint, its offsets slightly (alpha value) based on testing results from cross validation. Instead of optimizing for best fit to training data, it's optimizing for lowest cross validation scores. This is known as L2 regularisation. This model shrinks all of the correlated features.  

* Good to use when the data has multicollinearity (when there is correlation between features 
* Relationship is linear, little outliers, and independence

The Ridge model on sklearn has build in cross validation to find the best alpha value.

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV):
* alphas = (default 0.1, 1.0, 10.0) ndarray of alpha values, larger values mean stronger regularization 
* scoring = (default NONE) [Here](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules) for all the types of scoring methods
* cv = (default NONE) cross validation method. int is number of folds

<sub>[Read more](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf)<sub>

---------

#### [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

`from sklearn.linear_model import Lasso`

Very similar to RidgeRegression, however Lasso is able to reduce (all the way to 0) the infulence of features that dont provide any valuable insight for the target. However. is not as good when all the variables are usefull. This model picks one of the correlated features and removes the others. This is known as L1 regularisation. 

* Can be used to help feature selection 
* Works well when there are a lot of useless features 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso):
* alpha = (default 1) refer to ridge
* random_state = (default none) int for seeding the random number generator

<sub>Also see [Multi-task Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso)<sub>

--------



#### [Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

`from sklearn.linear_model import ElasticNet`

Is a combination of both RidgeRegression and Lasso, combining both L1 and L2 regularisation. It preforms cross validation to find the best mix of both RidgeRegression and Lasso. This model is able to both shrink features aswell as remove them from the equation

* Strong when handling correlation between features 
* Good If you are unsure if you need L1 or L2 regularisation

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)
* alpha = (default 1) refer to ridge 
* l1_ratio = (default 0.5) where 0 is more l2 favoured and 1 is l1 favoured 

<sub>Also see [Multi-task Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)<sub>

--------

#### [Bayesian](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)

`from sklean.linear_model import BayesianRidge`

Baysian allows you to represent uncertainty. This approach is able to create trends based off old data and update its trends based on new data, assuming that the model is not perfect and needs to adjust for error. As the amount of data increase toward infinite the model gets closer to LinearRegression. 

* Good when the target values will not fall within the same x values as your training data. 
* For example in time series when you are predicting future targets
* Or perhaps for sports metrics models predicting games played after a long break (COVID) where there is expected to be uncertainty

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge):
* Using the correct weights for the lambda and alpha values require optimization and regularization. Refer [Here](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression) for a user guide 

![](https://lh3.googleusercontent.com/proxy/3yZOQMS8YPbECTOJDFZEylgvdh_t0COry7uAbmSJZjpA1VsuLowBOQwNSPDB0_ZhhiU3s8h_NOcxhsLa-QYPb87cKSAoyogAygJjPT990mTUf4ttO0g)


<sub> [Reference](http://www.l4labs.soton.ac.uk/tutorials/excel/10e11.htm) <sub>


    
------

#### [Stochastic Gradient Decent (Regression)](https://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd)

`from sklearn.linear_model import SGDRegressor`

Used to fit a linear relationship between varibales, major benifit to this model is that it is very efficient. Meaning that it is best suited for large amounts of data (more than 10,000 trianing samples/ features). The model itself does not differ from ridge/lasso/elastic net, this is just a technique to optimize the training process. 

* Strong at handling large amounts of data
* Weak when dealing with feature scaling 
* Sensitive to outliers in the data

This is because the way the method speeds up the process of gradient decent is that it only takes one point in the data per step towards the least error. If that one point selected at random is an outlier, then you are going to have unwanted shift in the training process. However, with large amounts of data this shouldn't be a big problem as you will eventually reach somewhere close to the minimum loss. 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor):
* shuffle = (default True) shuffles the data before training, this is very important, if false make sure you do this manually 
* loss = (default 'squared_loss') used to dermine which loss function to use. some are better with outliers, others ingore error. 
* penalty = (default 'l2') chooses regulaization term
* learning_rate = (default invscaling) can be 'constant', 'optimal', invscaling', or 'adaptive'. 
* max_iter = (default 1000) number of itterations 

<sub>For Classification see [Stochastic Gradient Descent (Classifier)](#Stochastic-Gradient-Descent-(Classifier))<sub>


-------------

#### [Support Vector Regression](https://scikit-learn.org/stable/modules/svm.html#regression)

`from sklean.svm import SVR`

Support Vectors Machines are more commonly used for classification problems. Refer to [Suppot Vector Classification](#Suppot-Vector-Classification) for more information and diagrams. However, this concept can be extended to regression problems. There are a few different types of SVR. LinearSVR is the fastest, but only considers a linear kernel. Also NuSVR which has a slightly different formulation than SVR, refer [here](https://scikit-learn.org/stable/modules/svm.html#svm-implementation-details) if you would like to learn more. 

--------

### Classification
Problems where you are trying to predict if an outcome will happen or not (a true of false outcome). For example, if a team will win or lose a match. 

#### [LogisticRegression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)

`from sklearn.linear_model import LogisticRegression`

Even though this model is called a regression model, it is used to predict the probability of an outcome. It does this by matching an S curve (logistic function) between the 2 outcomes. 


* Good for probability scores 
* Can handle multi-collinearity with different types of regularization 
* Weak with large number of categorical features 
* One negative thing about this method is that it assumes linearity between the feature and target. Meaning that as the feature value increases, the target proability goes one direction. In practice it is often not that clear and simple

For example perhaps we are using age to predict  ablitly to complete a cognitive task, this data would show that as you get older (towards an adult) your probability of completing the task will increase. The model would fit to this relationship. However, it would not be able to caputre the fact that as people become more elderly, the proability would be going back down. This is an example where the feature is not linear towards the target.

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression):

* solver = (default 'lbfgs') this parameters defines the type of optimization to use, consider reading more about these to best fit your data. 
* penalty = (default 'l2') changes the type of regulaization that is used, this needs to compatible with the selected solver 
* C = (default 1.0) float value that determines the strength of regularization, smaller numbers mean stronger.

![](https://miro.medium.com/max/1200/0*Uuzp5aXD5zytA-F3.png) <sub>[Reference](https://medium.com/analytics-vidhya/logistic-regression-8f037d180d6f)<sub>

-------

#### [Stochastic Gradient Descent (Classifier)](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

`from sklearn.linear_model import SGDClassifier`

Used in the same way as [Stochastic Gradient Decent (Regression)](#Stochastic-Gradient-Decent-(Regression)) but for classicication problems.
 
* Good for handling large amounts of data 
* Requires a lot of parameter tuning 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier)

* shuffle = (default True) shuffles the data before training, this is very important, if false make sure you do this manually
* loss = (default 'squared_loss') used to dermine which loss function to use. some are better with outliers, others ingore error.
* penalty = (default 'l2') chooses regulaization term
* learning_rate = (default invscaling) can be 'constant', 'optimal', invscaling', or 'adaptive'.
* max_iter = (default 1000) number of itterations

---------

#### [Linear Discriminant Analysis](https://scikit-learn.org/stable/modules/lda_qda.html)

`from sklearn.discriminant_analysis import LinearDiscriminantAnalysis`

Creates linear boundaries between the outcome categories. This tries to maximize the seperation between the categories. This can handle multiple categorical outcomes. 

* Assumes that there is a Guassian relationships in the data (bell curved) 
* Also that each attribute has the same varriance
* Weak with few catgeorical variables 

Refer [here](#Quadratic-Discriminant-Analysis) for a visual to help understand this concept.

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis):
* solver = (default 'svd') good for large number of features, try other solvers if low number of features. 
* shrinkage = (default None) can be 'auto' or a float 0-1 
* n_components = (defualt None) number of components for reducing dimensionality 

-------

#### [Quadratic Discriminant Analysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)

`from sklearn.discrimiant_analsys import QuadraticDiscriminantAnalysis`

Creates quadratic decision boundaries between the outcome categories. 

* Assumes that there is a Guassian relationships in the data (bell curved)

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html#sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis):
* prior = (defualt None) ndarray of shape, or is inferred by the trianing data. 
* reg_param = (default False) float to set regularization. 



This diagram from the scikit website helps show how disciminant analysis works
![](https://scikit-learn.org/stable/_images/sphx_glr_plot_lda_qda_0011.png)
<sub>[Reference](https://scikit-learn.org/stable/modules/lda_qda.html)<sub>

-------

#### [Naive Bayes (Mulrinomial)](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)

`from sklearn.naive_bayes import MultinomialNB`

This model is used to estimate the classification probabily based of frequency found in test data. Think spam emails as an example. By comparing the contents of an unknown incoming email with exsiting frequency in the test data, it can predict the chances of being in both groups. It then classifies it as the one with high probability. 

* Very simple model that often performs well in practice (low-bias, high-variance) 
* Used for text classification (discrete features)
* Treats all words the same regarless of order 
* Strong with low categories 
* Weak with multicollinearity 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB):
* Alpha = (defualt 1.0) This makes sure the model doesnt produce 0 values for words that are not found in one class (causes everything to be canceled out do to x*0). 

<sub>Other types of Naive Bayes: [Here](https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes)<sub>

-----------

#### [Suppot Vector Classification](https://scikit-learn.org/stable/modules/svm.html#classification)

`from sklearn.svm import SVC`

SVM's can be used for binary (success/failure) or multi-class classification (many possible outcomes). What this model does is it plots all the features in the same n-demensinal space and then tries to find the best plane to differentiate the classes. 

* Produces very good results when there is clear margin separation, lots of features compared to samples. 
* Not so good with large data sets because of long training time, or when there is overlapping in the target classes.
* Some examples of when SVC is used is in face detection, handwriting recognition, image classificaiton and bioinformatics. 

[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC):
* C = (default 1.0) float for regularization, between 0-1 
* kernel = (default 'rbf') used to specify different kernal types to be used (eg: 'linear','poly', 'sigmoid', ect)  
* other notible parameters include gamma, coef0, probability, and max_iter

This diagram helps show how the model works, but instead of just 2-D, it is n-D (n being the number of features)

![](https://scikit-learn.org/stable/_images/sphx_glr_plot_iris_svc_0011.png)

<sub>[Reference](https://scikit-learn.org/stable/modules/svm.html#classification)<sub> 

-----

References: 
   * [Scikit-learn](https://scikit-learn.org/stable/index.html)
   * [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)
   * Search results, stackoverflow, and wikipedia
