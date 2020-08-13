{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# - - Work in progress - - \n",
    "# Model Selection For Beginners\n",
    "---------------------------------------------------\n",
    "> ###### By: Graham Pinsent "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Purpose of this notebook: \n",
    "* Summary of machine learning models (from scikit-learn)\n",
    "* Focuses on the applications of the models\n",
    "* Reading through documentation about these models can often be very intimidating for beginners, especially when you do not have a strong math background. This is meant to be written for more people to understand the difference between models \n",
    "* I am not an expert by any means, some say one of the best ways to learn is by teaching. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Table of Contents \n",
    "\n",
    "* [Regression](#Regression)\n",
    "  * Linear\n",
    "   * [LinearRegression](#LinearRegression)\n",
    "   * [RidgeRegression](#RidgeRegression)\n",
    "   * [Lasso](#Lasso)\n",
    "   * [Elastic-Net](#Elastic-Net)\n",
    "   * [Bayesian](#Bayesian)\n",
    "* [Classification](#Classification)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Regression \n",
    "Regression problems invole predicting a value that is continuous. For example predicting the weight of something, or monthly sales. There is a range of possible outcomes. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### [LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`from sklearn.linear_model import LinearRegression`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Also known as: Ordinary Least Squares. Line of best fit where the average distance to each data point is the lowest possible with a straight line. \n",
    "\n",
    "* Low bias \n",
    "* High Varriance \n",
    "* Highly sensitive to random y_train errors (outliers in the target data)\n",
    "* Assumes no correlation between features, which is rare.\n",
    "\n",
    "This is a simple, quick to run, model that will not pick up on complex relationships between features and the target. However, when there is not a lot of data or there is large noise in the data, this can often produce better results than more complex models. \n",
    "\n",
    "[Prarameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html):\n",
    "* fit_intercept = (defualt-True) if false, the y-intercept=0\n",
    "* normalize = (default-FALSE) if true, X will be subtracted by the mean and divided by l2-norm. \n",
    "* copy_X = (default-True) if true, X is copied not overwritten\n",
    "-------------\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### [RidgeRegression](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`from sklean.linear_model import RidgeCV`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is the similar to linear regression but has a way to reduce overfitting to training data. Instead of getting the lowest mean to each datapoint, its offsets slightly (alpha value) based on testing results from cross validation. Instead of optimizing for best fit to training data, it's optimizing for lowest cross validation scores. This is known as L2 regularisation. This model shrinks all of the correlated features.  \n",
    "\n",
    "* Good to use when the data has multicollinearity (when there is correlation between features \n",
    "* Relationship is linear, little outliers, and independence\n",
    "\n",
    "The Ridge model on sklearn has build in cross validation to find the best alpha value.\n",
    "\n",
    "[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV):\n",
    "* alphas = (default 0.1, 1.0, 10.0) ndarray of alpha values, larger values mean stronger regularization \n",
    "* scoring = (default NONE) [Here](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules) for all the types of scoring methods\n",
    "* cv = (default NONE) cross validation method. int is number of folds\n",
    "\n",
    "[Read more](https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/NCSS/Ridge_Regression.pdf)\n",
    "\n",
    "---------"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### [Lasso](https://scikit-learn.org/stable/modules/linear_model.html#lasso)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`from sklearn.linear_model import Lasso`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Very similar to RidgeRegression, however Lasso is able to reduce (all the way to 0) the infulence of features that dont provide any valuable insight for the target. However. is not as good when all the variables are usefull. This model picks one of the correlated features and removes the others. This is known as L1 regularisation. \n",
    "\n",
    "* Can be used to help feature selection \n",
    "* Works well when there are a lot of useless features \n",
    "\n",
    "[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso):\n",
    "* alpha = (default 1) refer to ridge\n",
    "* random_state = (default none) int for seeding the random number generator\n",
    "\n",
    "Also see [Multi-task Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLasso.html#sklearn.linear_model.MultiTaskLasso)\n",
    "\n",
    "--------\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### [Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`from sklearn.linear_model import ElasticNet`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Is a combination of both RidgeRegression and Lasso, combining both L1 and L2 regularisation. It preforms cross validation to find the best mix of both RidgeRegression and Lasso. This model is able to both shrink features aswell as remove them from the equation\n",
    "\n",
    "* Strong when handling correlation between features \n",
    "* Good If you are unsure if you need L1 or L2 regularisation\n",
    "\n",
    "[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)\n",
    "* alpha = (default 1) refer to ridge \n",
    "* l1_ratio = (default 0.5) where 0 is more l2 favoured and 1 is l1 favoured \n",
    "\n",
    "Also see [Multi-task Elastic-Net](https://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)\n",
    "\n",
    "--------"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "##### [Bayesian](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`from sklean.linear_model import BayesianRidge`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Baysian allows you to represent uncertainty. This approach is able to create trends based off old data and update its trends based on new data, assuming that the model is not perfect and needs to adjust for error. As the amount of data increase toward infinite the model gets closer to LinearRegression. \n",
    "\n",
    "* Good when the target values will not fall within the same x values as your training data. \n",
    "* For example in time series when you are predicting future targets\n",
    "* Or perhaps for sports metrics models predicting games played after a long break (COVID) where there is expected to be uncertainty\n",
    "\n",
    "[Parameters](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge):\n",
    "* Using the correct weights for the lambda and alpha values require optimization and regularization. Refer [Here](https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression) for a user guide \n",
    "\n",
    "\n",
    "------"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Classification\n",
    "Problems where you are trying to predict if an outcome will happen or not (a true of false outcome). For example, if a team will win or lose a match. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "References: \n",
    "   * [Scikit-learn](https://scikit-learn.org/stable/index.html)\n",
    "   * [StatQuest](https://www.youtube.com/channel/UCtYLUTtgS3k1Fg4y5tAhLbw)\n",
    "   * Search results, stackoverflow, and wikipedia"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
