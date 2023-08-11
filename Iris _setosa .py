#!/usr/bin/env python
# coding: utf-8

# # The Iris setosa

# The Iris flower data set or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis.
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.
# 
# Here's a picture of the three different Iris types:

# In[2]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300, height=300)


# In[3]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300, height=300)


# In[4]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300, height=300)


# The iris dataset contains measurements for 150 iris flowers from three different species.
# 
# The three classes in the Iris dataset:
# 
# Iris-setosa (n=50)
# Iris-versicolor (n=50)
# Iris-virginica (n=50)
# The four features of the Iris dataset:
# 
# sepal length in cm
# sepal width in cm
# petal length in cm
# petal width in cm

# # Get the data

# Get the data
# *Use seaborn to get the iris data by using: iris = sns.load_dataset('iris') 

# In[5]:


import seaborn as sns
iris=sns.load_dataset('iris')


# # Exploratory Data Analysis

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


iris.head()


# We Create a pairplot of the data set. Which flower species seems to be the most separable.

# In[8]:


sns.pairplot(iris,hue='species',palette='Dark2')


# We Create a kde plot of sepal_length versus sepal width for setosa species of flower.

# In[9]:


setosa = iris[iris['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)


# # Train Test Split
# ** Split your data into a training set and a testing set.**

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X=iris.drop('species',axis=1)
y=iris['species']
X_train,X_test,y_train ,y_test= train_test_split(X, y, test_size=0.3,random_state=101)


# # Train a Model
# Now its time to train a Support Vector Machine Classifier.

# In[12]:


from sklearn.svm import SVC


# In[13]:


model = SVC()


# In[14]:


model.fit(X_train,y_train)


# # Model Evaluation
# Now get predictions from the model and create a confusion matrix and a classification report.

# In[15]:


predictions = model.predict(X_test)


# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(y_test,predictions))


# In[19]:


print(confusion_matrix(y_test,predictions))


# In[20]:


print(classification_report(y_test,predictions))


# Wow! We can notice that our model was pretty good!

# In[ ]:




