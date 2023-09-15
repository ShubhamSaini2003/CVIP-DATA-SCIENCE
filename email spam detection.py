#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
import seaborn as sbs


# In[6]:


gtav=pd.read_csv(r"C:\Users\iamsh\Downloads\emails.csv")


# In[7]:


gtav


# # Drop all columns except Email No. and the

# In[8]:


gtav=gtav.iloc[:,:2]


# In[9]:


gtav


# In[10]:


gtav.shape


# In[11]:


gtav.dropna()


# In[12]:


gtav.describe()


# In[ ]:





# # Data Preprocessing

# In[13]:


gtav.isnull().sum()


# In[14]:


gtav.isnull().values.any()


# In[15]:


gtav.the.value_counts()


# # Data and output

# In[16]:


x=gtav['Email No.']


# In[17]:


y=gtav['the']


# In[18]:


y


# In[19]:


gtav.head(10)


# # Data visualization

# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x=cv.fit_transform(x)


# In[21]:


x.toarray()


# # dividing Traing and testting data

# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=1)


# # Model selection

# In[24]:


from sklearn.naive_bayes import MultinomialNB


# In[25]:


gnb=MultinomialNB()
gnb.fit(xtrain,ytrain)


# In[26]:


gnb.score(xtest,ytest)


# In[27]:


from sklearn.naive_bayes import BernoulliNB
bnb=BernoulliNB()
bnb.fit(xtrain,ytrain)


# In[28]:


bnb.score(xtest,ytest)


# # K Fold validation

# In[29]:


from sklearn.model_selection import cross_val_score
cv_score=cross_val_score(gnb,x,y,cv=10)


# In[30]:


cv_score


# In[31]:


cv_score.mean()


# In[ ]:




