
# coding: utf-8

# # Project 2: Supervised Learning
# ### Building a Student Intervention System

# ## 1. Classification vs Regression
# 
# Your goal is to identify students who might need early intervention - which type of supervised machine learning problem is this, classification or regression? Why?

# ## 2. Exploring the Data
# 
# Let's go ahead and read in the student dataset first.
# 
# _To execute a code cell, click inside it and press **Shift+Enter**._

# In[85]:
from __future__ import division
# Import libraries
import numpy as np
import pandas as pd
from sklearn import cross_validation
import matplotlib
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold,SelectPercentile, f_classif


# In[86]:

# Read student data
student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"
# Note: The last column 'passed' is the target/label, all other are feature columns


# Now, can you find out the following facts about the dataset?
# - Total number of students
# - Number of students who passed
# - Number of students who failed
# - Graduation rate of the class (%)
# - Number of features
# 
# _Use the code block below to compute these values. Instructions/steps are marked using **TODO**s._

# In[87]:

# TODO: Compute desired values - replace each '?' with an appropriate expression/function call
n_students = student_data.shape[0]
n_features = student_data.shape[1]-1
n_passed = student_data.loc[student_data['passed']=='yes'].shape[0]
n_failed = n_students-n_passed
grad_rate = 100*n_passed/n_students
print "Total number of students: {}".format(n_students)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Number of features: {}".format(n_features)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)


# ## 3. Preparing the Data
# In this section, we will prepare the data for modeling, training and testing.
# 
# ### Identify feature and target columns
# It is often the case that the data you obtain contains non-numeric features. This can be a problem, as most machine learning algorithms expect numeric data to perform computations with.
# 
# Let's first separate our data into feature and target columns, and see if any features are non-numeric.<br/>
# **Note**: For this dataset, the last column (`'passed'`) is the target or label we are trying to predict.

# In[88]:

# Extract feature (X) and target (y) columns
feature_cols = list(student_data.columns[:-1])  # all columns but last are features
target_col = student_data.columns[-1]  # last column is the target/label
print "Feature column(s):-\n{}".format(feature_cols)
print "Target column: {}".format(target_col)

X_all = student_data[feature_cols]  # feature values for all students
y_all = student_data[target_col]  # corresponding targets/labels
print "\nFeature values:-"
print X_all.head()  # print the first 5 rows


# ### Preprocess feature columns
# 
# As you can see, there are several non-numeric columns that need to be converted! Many of them are simply `yes`/`no`, e.g. `internet`. These can be reasonably converted into `1`/`0` (binary) values.
# 
# Other columns, like `Mjob` and `Fjob`, have more than two values, and are known as _categorical variables_. The recommended way to handle such a column is to create as many columns as possible values (e.g. `Fjob_teacher`, `Fjob_other`, `Fjob_services`, etc.), and assign a `1` to one of them and `0` to all others.
# 
# These generated columns are sometimes called _dummy variables_, and we will use the [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) function to perform this transformation.

# In[89]:

# Preprocess feature columns
def preprocess_features(X):
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX

X_all = preprocess_features(X_all)
print "Processed feature columns ({}):-\n{}".format(len(X_all.columns), list(X_all.columns))
print X_all.head()


#selector = VarianceThreshold(0.2)
#X_transform=selector.fit_transform(X_all)
#cols=X_all.columns[selector.get_support()]
#print cols, cols.size
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X_all, y_all)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
X_indices = np.arange(X_all.shape[-1])
plt.bar(X_indices , scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)', color='g')
plt.show


