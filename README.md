# Machine Learning Projects   [![python](https://camo.githubusercontent.com/352488c0cbba0e8f6da11ae0761444dd0c93489c/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f707974686f6e2d322e372d626c75652e737667)](https://www.python.org/download/releases/2.7/) [![scikit-learn](https://camo.githubusercontent.com/9f0ed32d05350afa18a801573e4da7f4a240e181/68747470733a2f2f62616467652e667572792e696f2f70792f7363696b69742d6c6561726e2e737667)](https://badge.fury.io/py/scikit-learn)

Overview
--------

This repo includes projects done as part of Udacity's Machine Learning
NanoDegree program. Also included are support libraries developed by me, as part
of this program.

Dependency
----------

-   Python 2.7

-   Numpy 1.10.1

-   Scikit-Learn 0.17

-   Matplotlib 1.5.1

-   Seaborn 0.7.0

Projects
--------

### Supervised Learning


#### Boston Housing Prices
This project uses Boston Housing Data to predict price of a house, given its other features. First, a Cross Validation is performed, and then Decision Tree is used with Grid Search to create a price estimation model. Different training sizes were evaluated, and variance of prediction error was ploted to select best training size. Best results were found when **Depth of Tree was 5**.

#### Student Intervention

A classification model is built, with minimal computation cost, to identify students that require intervention to pass their class. This model uses Student's current and past academic records, and details about their life. Such details include informaiton about their schedule and their parents.

Support Vector Classifiction, Bagging, and Boosting algorithms are used to evaluate different models, based on time and memory efficiency, and accuracy scores.

Finally, AdaBoost was selected and tuned for **best accuracy score of 0.83**

### Unsupervised Learning
#### Customer Segments
A wholesale distributor wants to find the best schedule delivery of products to
its customers (retail shops). First task is to idenitfy different types of
retail shops, so that delivery methods could be tailored for each group. Second
task is to model A/B tests to improve delivery satisfaction, and hence sales. 

Initial exploratory data analysis suggested some trends, that are confirmed by
Primary Component Analysis (PCA). And then Independent Component Analysis (ICA)
suggests that volume is the primary differentiator.

Then K-Means and Gaussian Mixed Model (GMM) algorithms are applied to identify
segments in retail shops. On analysis of these 2 models, GMM is found to be more
suitable for given problem.

GMM model is used to label the data, and prediction scenarios are suggested.
Also suggested are A/B test strategies to improve delivery satisfaction for
retail shops.
