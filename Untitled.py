#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns





# In[2]:


df= pd.read_csv(r"C:\Users\iamsh\Downloads\Breast_cancer_data.csv",encoding=('ISO-8859-1'),low_memory =False)


# In[3]:


df.head(1)


# In[4]:


df.shape


# # Checking Null Values
# 

# In[52]:


print("\nNull Values:\n", df.isnull().sum())


# # Checking Missing Values
# 

# In[53]:


print("\nMissing Values:\n", df.isna().sum())


# # Information of data
# 

# In[54]:


df.info()


# # Statistical Description of Data
# 

# In[55]:


df.describe()


# # Extracting Mean, Squared Error and Worst Features with Diagnosis
# 

# In[56]:


df_mean = df[df.columns[:11]]
df_se = df.drop(df.columns[1:11], axis=1);
df_se = df_se.drop(df_se.columns[11:], axis=1)
df_worst = df.drop(df.columns[1:21], axis=1)


# Count Based On Diagnosis
# 

# In[57]:


df.diagnosis.value_counts()


# In[58]:


df.diagnosis.value_counts() \
    .plot(kind="bar", width=0.1, color=["lightgreen", "cornflowerblue"], legend=1, figsize=(8, 5))
plt.xlabel("(0 =  Benign) (1 =Malignant)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(fontsize=12);
plt.yticks(fontsize=12)
plt.legend([" Benign"], fontsize=12)
plt.show()


# Correlation with diagnosis:

# In[59]:


def corrwithdia(dfx):
    import matplotlib.pyplot as plt
    name = str([x for x in globals() if globals()[x] is dfx][0])
    if name == 'df':
        x = "All"
    elif name == 'df_mean':
        x = "Mean"
    elif name == 'df_se':
        x = "Squared Error"
    elif name == 'df_worst':
        x = "Worst"
    plt.figure(figsize=(20, 8))
    dfx.drop('diagnosis', axis=1).corrwith(dfx.diagnosis).plot(kind='bar', grid=True, title="Correlation of {} Features with Diagnosis".format(x), color="cornflowerblue");


# correlation of Mean Features with Diagnosis
# 

# In[60]:


corrwithdia(df_mean)


# Correlation of Squared Error Features with Diagnosis
# 

# In[61]:


corrwithdia(df_se)


# Correlation of Worst Features with Diagnosis
# 

# In[62]:


corrwithdia(df_worst)


# Extracting Mean, Squared Error and Worst Features

# In[63]:


df_mean_cols = list(df.columns[1:11])
df_se_cols = list(df.columns[11:21])
df_worst_cols = list(df.columns[21:])


# Split into two Parts Based on Diagnosis

# In[64]:


dfM = df[df['diagnosis'] == 1]
dfB = df[df['diagnosis'] == 0]


# Nucleus Mean Features vs Diagnosis

# In[65]:


plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.figure
    binwidth = (max(df[df_mean_cols[idx]]) - min(df[df_mean_cols[idx]])) / 50
    ax.hist([dfM[df_mean_cols[idx]], dfB[df_mean_cols[idx]]],
            bins=np.arange(min(df[df_mean_cols[idx]]), max(df[df_mean_cols[idx]]) + binwidth, binwidth), alpha=0.5,
            stacked=True, label=['M', 'B'], color=['b', 'g'])
    ax.legend(loc='upper right')
    ax.set_title(df_mean_cols[idx])
plt.tight_layout()
plt.show()


# Nucleus Squared Error Features vs Diagnosis

# In[66]:


plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.figure
    binwidth = (max(df[df_se_cols[idx]]) - min(df[df_se_cols[idx]])) / 50
    ax.hist([dfM[df_se_cols[idx]], dfB[df_se_cols[idx]]],
            bins=np.arange(min(df[df_se_cols[idx]]), max(df[df_se_cols[idx]]) + binwidth, binwidth), alpha=0.5,
            stacked=True, label=['M', 'B'], color=['b', 'g'])
    ax.legend(loc='upper right')
    ax.set_title(df_se_cols[idx])
plt.tight_layout()
plt.show()


# Nucleus Worst Features vs Diagnosis

# In[67]:


plt.rcParams.update({'font.size': 8})
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(8, 10))
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.figure
    binwidth = (max(df[df_worst_cols[idx]]) - min(df[df_worst_cols[idx]])) / 50
    ax.hist([dfM[df_worst_cols[idx]], dfB[df_worst_cols[idx]]],
            bins=np.arange(min(df[df_worst_cols[idx]]), max(df[df_worst_cols[idx]]) + binwidth, binwidth), alpha=0.5,
            stacked=True, label=['M', 'B'], color=['b', 'g'])
    ax.legend(loc='upper right')
    ax.set_title(df_worst_cols[idx])
plt.tight_layout()
plt.show()


# Checking Multicollinearity Between Different Features

# In[68]:


def pairplot(dfx):
    import seaborn as sns
    name = str([x for x in globals() if globals()[x] is dfx][0])
    if name == 'df_mean':
        x = "Mean"
    elif name == 'df_se':
        x = "Squared Error"
    elif name == 'df_worst':
        x = "Worst"
    sns.pairplot(data=dfx, hue='diagnosis', palette='crest', corner=True).fig.suptitle('Pairplot for {} Featrues'.format(x), fontsize = 20)


# # Mean Features:
# 
# 

# In[69]:


pairplot(df_mean)


# Squared Error Features:
# 
# 

# In[70]:


pairplot(df_se)


# Worst features:
# 
# 

# In[71]:


pairplot(df_worst)


# # Correlation Heartmap between Nucleus Feature

# In[72]:


corr_matrix = df.corr()  

mask = np.zeros_like(corr_matrix, dtype=np.bool_)
mask[np.triu_indices_from(corr_matrix)] = True

fig, ax = plt.subplots(figsize=(22, 10))
ax = sns.heatmap(corr_matrix, mask=mask, annot=True, linewidths=0.5, fmt=".2f", cmap="YlGn");
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5);
ax.set_title("Correlation Matrix Heatmap including all features");


# Outliers

# In[73]:


for column in df:
    plt.figure()
    df.boxplot([column])
    plt.show()


# Outlier removal

# In[74]:


from numpy import percentile
for column in df.columns:
    q25, q75 = percentile(df[column], 25), percentile(df[column], 75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    outliers = [x for x in df[column] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    outliers_removed = [x for x in df[column] if x >= lower and x <= upper]
    print('Non-outlier observations: %d' % len(outliers_removed))
    df = df[df[column] < upper]
    plt.figure()
    df.boxplot([column])
    plt.show()


# In[ ]:




