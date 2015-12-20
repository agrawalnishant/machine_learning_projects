import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import cross_validation
from sklearn.metrics import make_scorer
from sklearn import grid_search


def feature_min(boston,gs_estimator):
    # Load the boston dataset.
    boston = load_boston()
    X, y = boston['data'], boston['target']

    # We use the base estimator LassoCV since the L1 norm promotes sparsity of features.
    clf = LassoCV()

    # Set a minimum threshold of 0.25
    sfm = SelectFromModel(clf, threshold=0.5)
    sfm.fit(X, y)
    n_features = sfm.transform(X).shape[1]

    # Reset the threshold till the number of features equals two.
    # Note that the attribute can be set directly instead of repeatedly
    # fitting the metatransformer.
    while n_features > 2:
        sfm.threshold += 0.1
        X_transform = sfm.transform(X)
        n_features = X_transform.shape[1]

    # Plot the selected two features from X.

    feature1 = X_transform[:, 0]
    feature2 = X_transform[:, 1]
    print "features: ", boston.feature_names[sfm.get_support()]

    preds=np.zeros(506)
    preds=gs_estimator.predict(X)

    plt.figure()
    plt.title("Features selected from Boston using " "threshold %0.3f." % sfm.threshold)


    plt.scatter(feature1, y)

    """ Plot predicted value for  RM value in required vector. """
    plt.scatter(5.6090,20.968,color='green')

    """ Plot predicted prices for all data points, against RM. """
    plt.scatter(feature1,preds,color='red')

    plt.xlabel(boston.feature_names[sfm.get_support()][0])
    plt.ylabel("price")
    plt.xlim([np.min(feature1), np.max(feature1)])
    plt.ylim([np.min(y), np.max(y)])
    plt.show()

def load_data():
    """Load the Boston dataset."""
    
    boston = datasets.load_boston()
    return boston


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""
    
    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    
    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return np.sqrt(np.median(np.square(label - prediction)))


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""
    
    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target
    
    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()
    
    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    
    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################
    
    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    median_loss_scorer  = make_scorer(performance_metric, greater_is_better=False)
    
    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    gs_estimator= grid_search.GridSearchCV(regressor, parameters,scoring=median_loss_scorer)
    
    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    estmtr= gs_estimator.fit(X, y)
    print estmtr
    print "Best model parameter:  " + str(estmtr.best_params_)
    
    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = gs_estimator.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)

    return gs_estimator


def main():
    # Load data
    city_data = load_data()
   
    # Tune and predict Model
    gs_estimator=fit_predict_model(city_data)

    feature_min(city_data,gs_estimator)


main()



