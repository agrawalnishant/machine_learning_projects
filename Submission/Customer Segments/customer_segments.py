
# coding: utf-8

# # Creating Customer Segments

# In this project you, will analyze a dataset containing annual spending amounts for internal structure, to understand the variation in the different types of customers that a wholesale distributor interacts with.
# 
# Instructions:
# 
# - Run each code block below by pressing **Shift+Enter**, making sure to implement any steps marked with a TODO.
# - Answer each question in the space provided by editing the blocks labeled "Answer:".
# - When you are done, submit the completed notebook (.ipynb) with all code blocks executed, as well as a .pdf version (File > Download as).

# In[48]:

# Import libraries: NumPy, pandas, matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tell iPython to include plots inline in the notebook
#get_ipython().magic(u'matplotlib inline')

import warnings
warnings.filterwarnings('ignore')

pd.options.display.float_format = '{:.5f}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Read dataset
data = pd.read_csv("wholesale-customers.csv")
num_features=data.shape[1]
num_data_points=data.shape[0]
print "Dataset has {} rows, {} columns".format(num_data_points,num_features)
print data.head()  # print the first 5 rows
print data.describe()

'''
TODOs:
1. Create 3-D plot for pca vectors.
2. Apply PCA on ICA-demixed-transformed data.
3. Draw elbow graph to identify k in k-means.
'''


# In[49]:

## Cleanup data, remove outliers.

f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.scatter(data.iloc[:,0],data.iloc[:,1])
ax1.set_xlabel('Fresh')
ax1.set_ylabel('Milk')

ax2.scatter(data.iloc[:,1],data.iloc[:,2])
ax2.set_xlabel('Milk')
ax2.set_ylabel('Grocery')


ax3.scatter(data.iloc[:,1],data.iloc[:,3])
ax3.set_xlabel('Milk')
ax3.set_ylabel('Frozen')

ax4.scatter(data.iloc[:,2],data.iloc[:,3])
ax4.set_xlabel('Grocery')
ax4.set_ylabel('Frozen')


ax5.scatter(data.iloc[:,3],data.iloc[:,4])
ax5.set_xlabel('Frozen')
ax5.set_ylabel('Detergents and Paper')


ax6.scatter(data.iloc[:,4],data.iloc[:,5])
ax6.set_xlabel('Detergents and Paper')
ax6.set_ylabel('Delicatessen')

fig = plt.gcf()
fig.set_size_inches(12, 12)
#fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()



print "-"*100
print "Histogram of spending on specific product types."
print "-"*100


f, ((axis1, axis2), (axis3, axis4), (axis5, axis6)) = plt.subplots(3, 2)

### Visualize data spread.
colormap = np.array(['r', 'g', 'b','c','m','y'])
f.axes[0].hist(data.iloc[:,0],bins=20,color=colormap[0])
f.axes[0].set_xlabel(data.columns.values[0]);

f.axes[1].hist(data.iloc[:,1],bins=20,color=colormap[1])
f.axes[1].set_xlabel(data.columns.values[1]);


f.axes[2].hist(data.iloc[:,2],bins=20,color=colormap[2])
f.axes[2].set_xlabel(data.columns.values[2]);

f.axes[3].hist(data.iloc[:,3],bins=20,color=colormap[3])
f.axes[3].set_xlabel(data.columns.values[3]);

f.axes[4].hist(data.iloc[:,4],bins=20,color=colormap[4])
f.axes[4].set_xlabel(data.columns.values[4]);

f.axes[5].hist(data.iloc[:,5],bins=20,color=colormap[5])
f.axes[5].set_xlabel(data.columns.values[5]);

fig = plt.gcf()
fig.set_size_inches(12, 12)
fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()


# In[50]:

## Cleaning outliers could be useful, since it would remove noise which is more prevelent in low-variance components.


cleaned_data=data.copy(deep=True)

cleaned_data=cleaned_data[cleaned_data['Fresh']<60000]
cleaned_data=cleaned_data[cleaned_data['Milk']<50000]
cleaned_data=cleaned_data[cleaned_data['Grocery']<50000]
cleaned_data=cleaned_data[cleaned_data['Frozen']<30000]
cleaned_data=cleaned_data[cleaned_data['Detergents_Paper']<20000]
cleaned_data=cleaned_data[cleaned_data['Delicatessen']<20000]

# Removed scaling since units are same, and feature-wise expenses are part of same expense, i.e. belong to same part.
#from sklearn import preprocessing
#cleaned_data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']] = cleaned_data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']].apply(lambda x: preprocessing.StandardScaler().fit_transform(x))

print '-'*100
print " Cleaned, centered, and normalized data."
print '-'*100
f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.scatter(cleaned_data.iloc[:,0],cleaned_data.iloc[:,1])
ax1.set_xlabel('Fresh')
ax1.set_ylabel('Milk')

ax2.scatter(cleaned_data.iloc[:,1],cleaned_data.iloc[:,2])
ax2.set_xlabel('Milk')
ax2.set_ylabel('Grocery')

ax3.scatter(cleaned_data.iloc[:,1],cleaned_data.iloc[:,3])
ax3.set_xlabel('Milk')
ax3.set_ylabel('Frozen')

ax4.scatter(cleaned_data.iloc[:,2],cleaned_data.iloc[:,3])
ax4.set_xlabel('Grocery')
ax4.set_ylabel('Frozen')


ax5.scatter(cleaned_data.iloc[:,3],cleaned_data.iloc[:,4])
ax5.set_xlabel('Frozen')
ax5.set_ylabel('Detergents and Paper')


ax6.scatter(cleaned_data.iloc[:,4],cleaned_data.iloc[:,5])
ax6.set_xlabel('Detergents and Paper')
ax6.set_ylabel('Delicatessen')

fig = plt.gcf()
fig.set_size_inches(12, 12)
fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()
print " -> Fresh vs Milk, Milk vs Frozen, Grocery vs Frozen, and Detergent_paper vs Frozen all seem to have inverse relationship.\n"


print "-"*100
print "Histogram of spending on specific product types."
print "-"*100


f, ((axis1, axis2), (axis3, axis4), (axis5, axis6)) = plt.subplots(3, 2)

### Visualize data spread.
colormap = np.array(['r', 'g', 'b','c','m','y'])
f.axes[0].hist(cleaned_data.iloc[:,0],bins=20,color=colormap[0])
f.axes[0].set_xlabel(cleaned_data.columns.values[0]);

f.axes[1].hist(cleaned_data.iloc[:,1],bins=20,color=colormap[1])
f.axes[1].set_xlabel(cleaned_data.columns.values[1]);


f.axes[2].hist(cleaned_data.iloc[:,2],bins=20,color=colormap[2])
f.axes[2].set_xlabel(cleaned_data.columns.values[2]);

f.axes[3].hist(cleaned_data.iloc[:,3],bins=20,color=colormap[3])
f.axes[3].set_xlabel(cleaned_data.columns.values[3]);

f.axes[4].hist(cleaned_data.iloc[:,4],bins=20,color=colormap[4])
f.axes[4].set_xlabel(cleaned_data.columns.values[4]);

f.axes[5].hist(cleaned_data.iloc[:,5],bins=20,color=colormap[5])
f.axes[5].set_xlabel(cleaned_data.columns.values[5]);

fig = plt.gcf()
fig.set_size_inches(12, 12)
fig.set_size_inches(18.5, 10.5, forward=True)
plt.show()

print "-> Plots of Fresh, Milk, Grocery, and Frozen seems to have some similarity in shape and scale.\n"


# ##Feature Transformation

# **1)** In this section you will be using PCA and ICA to start to understand the structure of the data. Before doing any computations, what do you think will show up in your computations? List one or two ideas for what might show up as the first PCA dimensions, or what type of vectors will show up as ICA dimensions.

# Answer:
# 
# Idea 1. Based on data spread, first PCA would be either fresh, or it could be combination of milk and groceries.
# Second PCA could include Frozen and Detergent_Paper, and Third PCA could be delicatessen.
# 
# Idea 2. ICA could identify perishability as the differentiator in consumables / non-Delicatessen.

# ###PCA

# In[51]:

# TODO: Apply PCA with the same number of dimensions as variables in the dataset
# Using original data
from sklearn.decomposition import PCA
pca = PCA(n_components=num_features,whiten=True)
pca.fit(data)

# Print the components and the amount of variance in the data contained in each dimension
print data.columns.values
print pca.components_
print pca.explained_variance_ratio_


print "\n",'*'*5,"PCA on original data.",'*'*5,"\n"
pc_df=pd.DataFrame({"pca":pca.explained_variance_ratio_})
pc_cmf_df=np.cumsum(pc_df)
print '*'*5," Cumm variance:",'*'*5,"\n",pc_cmf_df
plt.plot(pc_cmf_df)
plt.ylabel('cumm. variance')
plt.xlabel('features')
plt.show()


# In[52]:

# TODO: Apply PCA with the same number of dimensions as variables in the dataset
# Using cleaned up data

from sklearn.decomposition import PCA
pca = PCA(n_components=num_features,whiten=True)
pca.fit(cleaned_data)

# Print the components and the amount of variance in the data contained in each dimension
print cleaned_data.columns.values
print pca.components_
print pca.explained_variance_ratio_


print "\n",'*'*5,"PCA on cleaned data.",'*'*5,"\n"
pc_df=pd.DataFrame({"pca":pca.explained_variance_ratio_})
pc_cmf_df=np.cumsum(pc_df)
print '*'*5," Cumm variance:",'*'*5,"\n",pc_cmf_df
plt.plot(pc_cmf_df)
plt.ylabel('Cumm. variance')


# In[53]:

''' Following function has been taken from Udacity Forum: 
https://discussions.udacity.com/t/
having-trouble-with-pca-and-ica-specifically-with-explaining-what-the-dimensions-mean/41890/12
'''

def biplot12(df):
    # Fit on 2 components
    pca = PCA(n_components=2, whiten=True).fit(df)
    
    # Plot transformed/projected data
    ax = pd.DataFrame(
        pca.transform(df),
        columns=['PC1', 'PC2']
    ).plot(kind='scatter', x='PC1', y='PC2', figsize=(10, 8), s=0.8)

    # Plot arrows and labels
    for i, (pc1, pc2) in enumerate(zip(pca.components_[0], pca.components_[1])):
        ax.arrow(0, 0, pc1, pc2, width=0.001, fc='orange', ec='orange')
        ax.annotate(df.columns[i], (pc1, pc2), size=12)

    return ax

print '-'*100
print "PC1 / PC2: Bi-plot of original data"
print '-'*100
ax = biplot12(data)
# Play around with the ranges for scaling the plot
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
plt.show()


print '-'*100
print "PC1 / PC2: Bi-plot of cleaned data."
print '-'*100

ax = biplot12(cleaned_data)
# Play around with the ranges for scaling the plot
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])


# In[54]:

# TODO: draw a 3-D plot ( or triplot :))
def biplot34(df):
    # Fit on 2 components
    pca = PCA(n_components=4, whiten=True).fit(df)
    
    # Plot transformed/projected data
    ax = pd.DataFrame(
        pca.transform(df),
        columns=['PC1', 'PC2','PC3','PC4']
    ).plot(kind='scatter', x='PC3', y='PC4', figsize=(10, 8), s=0.8)

    # Plot arrows and labels
    for i, (pc3, pc4) in enumerate(zip(pca.components_[2], pca.components_[3])):
        ax.arrow(0, 0, pc3, pc4, width=0.001, fc='orange', ec='orange')
        ax.annotate(df.columns[i], (pc3, pc4), size=12)

    return ax

print '-'*100
print "PC3 / PC4: biplot of cleaned data"
print '-'*100

ax = biplot34(cleaned_data)
# Play around with the ranges for scaling the plot
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])


# **2)** How quickly does the variance drop off by dimension? If you were to use PCA on this dataset, how many dimensions would you choose for your analysis? Why?

# Answer: Variance drops fast for first 2 dimentions, but then reduces slowly for remaining dimentions.
# Given the PCA variance graphs above, elbow is formed at 2nd PCA component, both for original data and scaled data. But since there are data points that have a some variance along multiple PCAs. 

# **3)** What do the dimensions seem to represent? How can you use this information?

# Answer:  PCA here can be used in 2 ways here: 1.) to identify similar customers. 2.)To find similar features. But target here is to find similar customers, and first 2 primary components seem to cover a most of variance. 
# 
# Then, first PCA dimention corresponds to a segment that spends mostly on Fresh and Frozen products.
# 
# Second PCA corresponds that spend mostly on Grocery, and significantly on Milk and Detergent_Paper in that order.
# 
# We can use this information in many ways: 
# 1.) To transform the data along these 2 PCA, and then find cluster of users using transformed data. But this may not be good approach, since PCA-transformed data might loose some information which could impact be useful for un-biased clustering. 
# 
# 2.) To do clustering independently, and then compare the results with those from PCA, to see if both these results are convergent of divergent.
# 
# 3.) We can use the results of PCA further components for supervised learning analysis - regression or classification.
# 
# 4.) We could also use K=2 and K=3 for k-mens clustering. Although value of K could depend on elbow in sum-of-square vs k plot.

# ###ICA

# In[55]:

# TODO: Fit an ICA model to the data
# Note: Adjust the data to have center at the origin first!
from sklearn.decomposition import FastICA
from sklearn import preprocessing

scaled_data=data.copy(deep=True)

#from sklearn import preprocessing
scaled_data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']] = scaled_data[['Fresh','Milk','Grocery','Frozen','Detergents_Paper','Delicatessen']].apply(lambda x: preprocessing.StandardScaler().fit_transform(x))

ica = FastICA(whiten=True,random_state=0)
transformed_data=ica.fit_transform(scaled_data)
#
# Print the independent components

print "\n"
print scaled_data.columns.values
print ica.components_

print "\n"
print preprocessing.StandardScaler().fit_transform(ica.components_)
#print "\n"
#print ica.mixing_


# **4)** For each vector in the ICA decomposition, write a sentence or two explaining what sort of object or property it corresponds to. What could these components be used for?

# Answer: 
# 
# ['Fresh' 		'Milk'		 'Grocery' 'Frozen'   'Detergents_Paper' 'Delicatessen']
# 
# A. [ 0.45910575  0.11611315  1.11498351  0.41769164 -0.48743545 	  -1.21961489] -- Delicacy, Grocery, small qty of everything.
# 
# B. [ 0.51224275 -0.02365303 -1.95603618  0.67015972  2.21983183 	  -0.03356863] -- Detergents_Paper, Grocery, Frozen, Fresh 
# 
# C. [ 0.23572202 -2.13321102  0.9452022   0.42088438 -0.6157666  	   0.03913673] -- Milk and Grocery, small qty of everything except delicacy. 
# 
# D. [-2.16830789  0.84785851  0.11414921  0.48780439 -0.53541034 	  -0.78091823]  -- Fresh, Milk, Delicacy 
# 
# E. [ 0.08837092  0.54640212 -0.09382264  0.21997389 -0.29311692		   1.97475038]  -- Delicacy, Milk 
# 
# F. [ 0.87286647  0.64649027 -0.12447609 -2.21651401 -0.28810251 	   0.02021463]  -- Frozen, fresh, some milk 
# 
# 
# -> A,B, and C seem to be some kind of grocery store or eatery.
# 
# -> D,E, and F dont buy much grocery, and seem to be specialized store, either bakery or chocolatier.
# 
# -> So this could mean the purchasing habbits of consumers. For example, consumers purchase milk from different sources.
# 

# ##Clustering
# 
# In this section you will choose either K Means clustering or Gaussian Mixed Models clustering, which implements expectation-maximization. Then you will sample elements from the clusters to understand their significance.

# ###Choose a Cluster Type
# 
# **5)** What are the advantages of using K Means clustering or Gaussian Mixture Models?

# Answer: 
# 
# 1. k-means is intuitive, and fast.
# 2. k-means can be computed and stored, for later application. This would allow quickly finding similarity. 

# **6)** Below is some starter code to help you visualize some cluster data. The visualization is based on [this demo](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html) from the sklearn documentation.

# In[56]:

# Import clustering modules
from sklearn.cluster import KMeans
from sklearn.mixture import GMM


# In[57]:

# TODO: First we reduce the data to two dimensions using PCA to capture variation

pca = PCA(n_components=2, whiten=True)

reduced_data = pca.fit_transform(cleaned_data)
print reduced_data[:10]  # print upto 10 elements


# In[58]:

# TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
# The visualizer below assumes your clustering object is named 'clusters'
from scipy.spatial.distance import cdist

clusters = KMeans(init='k-means++', n_clusters=3, n_init=5).fit(reduced_data)
print clusters

# Plot the decision boundary by building a mesh grid to populate a graph.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
hx = (x_max-x_min)/1000.
hy = (y_max-y_min)/1000.
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

# Obtain labels for each point in mesh. Use last trained model.
Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])


# In[59]:

# TODO: Find the centroids for KMeans or the cluster means for GMM 

centroids = clusters.cluster_centers_
print centroids



# In[60]:

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

print cleaned_data.columns.values
print centroids
print pca.inverse_transform(centroids)


# In[61]:

from matplotlib.colors import LogNorm
import matplotlib as mpl

n_classes=2
def make_ellipses(gmm, ax):
    for n, color in enumerate('rg'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        
        
classifiers = dict((covar_type, GMM(n_components=2,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])
#clf=GMM(n_components=3, covariance_type='full').fit(reduced_data)

n_classifiers = len(classifiers)

plt.figure(figsize=(3 * n_classifiers / 2, 6))
plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

for index, (name, classifier) in enumerate(classifiers.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    # classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
    #                              for i in xrange(n_classes)])

    # Train the other parameters using the EM algorithm.
    classifier.fit(reduced_data)

    h = plt.subplot(2, n_classifiers / 2, index + 1)
    make_ellipses(classifier, h)

    for n, color in enumerate('rg'):
        data = reduced_data
        plt.scatter(data[:, 0], data[:, 1], 0.8, color=color)
    

   

    plt.xticks(())
    plt.yticks(())
    plt.title(name)
    print cleaned_data.columns.values
    print classifier.sample
    print  pca.inverse_transform(classifier.means_)

plt.legend(loc='lower right', prop=dict(size=12))


plt.show()




# **7)** What are the central objects in each cluster? Describe them as customers.

# Answer: 
# As we can see above by inverse transformming the centroids, here is explaination of 3 clusters.
# Cluster 1 has a balanced consumption of everything.
# Cluster 2 has highest consumption of Fresh products.
# Cluster 3 mostly consumes Milk, Grocery, and Detergent_paper products.
# 

# ###Conclusions
# 
# ** 8)** Which of these techniques did you feel gave you the most insight into the data?

# Answer: All 3 techniques gave different information, and collating all 3 techniques gives confidence in solution.PCA gave more direct info on primary conponents, while ICA and Clustering gave insight into source and unlabeled similarity in data.

# **9)** How would you use that technique to help the company design new experiments?

# Answer: 

# In[62]:

#Need some guidance here.


# **10)** How would you use that data to help you predict future customer needs?

# Answer: 

# In[ ]:

#Need some guidance here.

