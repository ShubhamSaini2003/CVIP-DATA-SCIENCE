#!/usr/bin/env python
# coding: utf-8

# ## **IMPORTING THE REQUIRED DEPENDENCIES**
# 

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
from IPython.display import Audio
import librosa.display


# 

# ### **DATASET LOADING**

# 

# In[29]:


paths = []
labels = []
for dirname, _, filenames in os.walk("/kaggle/input/toronto-emotional-speech-set-tess"):
    for filename in filenames:
        paths.append(os.path.join(dirname, filename))
        label = filename.split('_')[-1]
        label = label.split('.')[0]
        labels.append(label.lower())
    if len(paths) == 2800:
        break

print("DATASET HAS BEEN LOADED")        
        
        


# 

# **DATA OBSERVATION**

# In[30]:


df = pd.DataFrame()
df['speech'] = paths
df['label'] = labels
df.head()


# In[31]:


labels[:10]


# ****

# **CREATING DATAFRAME**

# In[32]:


gtav= pd.DataFrame()
gtav['labels'] = labels
gtav['speech'] = paths


# 

# In[33]:


gtav.describe()


# In[34]:


gtav.head()


# In[35]:


gtav['speech'].value_counts()


# In[36]:


gtav['labels'].value_counts()


# ### VISUALIZING DATA
# **DEFINING WAVEPLOT FUNCTION**

# In[37]:


def wvplt(dt, sr, emtn):
    plt.figure(figsize=(12,6))
    plt.title(emtn, size = 18)
    librosa.display.waveshow(dt, sr=sr)
    plt.show()


# **DEFINING SPECTOGRAM**

# In[38]:


def sptm(dt, sr, emtn):
    x= librosa.stft(dt)
    x_db= librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(12,6))
    plt.title(emtn, size=20)
    librosa.display.specshow(x_db, sr=sr, x_axis='time',y_axis='hz')
    plt.colorbar()


# **HAPPY**

# In[39]:


emtn ='happy'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# In[ ]:





# **HAPPY**

# In[40]:


emtn ='happy'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# **PLESAENT SURPRISE**

# In[41]:


emtn ='ps'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# **ANGRY**

# In[42]:


emtn ='angry'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# **FEAR**

# In[43]:


emtn ='fear'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# **SAD**

# In[44]:


emtn ='sad'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# **DISGUST**

# In[45]:


emtn ='disgust'
path = np.array(df['speech'][df['label']== emtn])[1]
data,sam_rt = librosa.load(path)
wvplt(data,sam_rt,emtn)
sptm(data,sam_rt,emtn)


# In[46]:


def com_mfcc(filename):
    y,sr=librosa.load(filename, duration=3, offset=0.5)
    mfccs=librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    avg_mfccs= np.mean(mfccs.T, axis=0)
    return avg_mfccs


# In[47]:


com_mfcc(df['speech'][0])


# In[48]:


c_mfcc=df['speech'].apply(lambda x: com_mfcc(x))
c_mfcc


# In[49]:


c_x=[x for x in c_mfcc]
c_x=np.array(c_x)
c_x.shape


# In[50]:


c_x=np.expand_dims(c_x,-1)
c_x.shape


# 

# #### MODEL TRAINIING AND PREDICTION

# 

# In[51]:


from sklearn.preprocessing import OneHotEncoder
encd = OneHotEncoder()
c_y=encd.fit_transform(dtfr[['labels']])
c_y=c_y.toarray()


# In[52]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(c_x,c_y,test_size=0.20, random_state=30)


# In[53]:


from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model= Sequential([
    LSTM(256, return_sequences=False, input_shape=(40,1)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(7, activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary


# **TRAINING THE MODEL**

# In[54]:


P=model.fit(x_train,y_train,validation_split=0.2,epochs=23, batch_size=64,shuffle=True)


# In[ ]:





# In[56]:


model.evaluate(x_train, y_train)


# In[57]:


Y_P=model.predict(x_test, batch_size=6)


# In[58]:


accuracy_s=model.evaluate(x_test,y_test)


# **GRAPH PLOT**

# In[ ]:





# In[ ]:




