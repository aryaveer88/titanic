
# coding: utf-8

# In[1]:

import random 


# In[5]:

a = random.randint(0,100)


# In[6]:

a


# In[7]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from sklearn import datasets 
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


# In[8]:

boston_dataset = datasets.load_boston()
X_full = boston_dataset.data
Y = boston_dataset.target
print X_full.shape 
print Y.shape


# In[9]:

print boston_dataset.DESCR


# In[10]:

selector = SelectKBest(f_regression, k=1)
selector.fit(X_full, Y)
X= X_full[:, selector.get_support()]
print X.shape


# In[11]:

plt.scatter(X,Y,color = 'red')


# In[12]:

regressor = LinearRegression(normalize=True)


# In[13]:

regressor.fit(X, Y)
plt.scatter(X,Y, color='black')
plt.plot(X, regressor.predict(X), color='blue', linewidth=3)
plt.show()


# In[17]:

regressor = RandomForestRegressor()
regressor.fit(X,Y)
plt.scatter(X,Y, color='blue')
plt.scatter(X, regressor.predict(X), color='red', linewidth=3)
plt.show()

