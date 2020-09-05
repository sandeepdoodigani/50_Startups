#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[3]:


import pandas as pd
import io
import requests
url="https://raw.githubusercontent.com/LakshmiPanguluri/Linear_Multiple_Regression/master/50_Startups.csv"
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))


# In[4]:


df.head()


# In[5]:


mean_dfrd=df["R&D Spend"].mean()
mean_dfrd


# In[6]:


df.RDSpend=df["R&D Spend"].fillna(mean_dfrd)


# In[7]:


df


# In[10]:


reg=linear_model.LinearRegression()
reg.fit(df[["R&D Spend",'Administration','Marketing Spend']],df.Profit)


# ### reg.coef_

# In[11]:


reg.predict([[542,51743,0]])


# In[12]:


State=pd.get_dummies(df['State'],drop_first=True)


# In[13]:


State


# In[14]:


df=pd.concat([df,State],axis=1)


# In[15]:


df


# In[20]:


reg3=linear_model.LinearRegression()
reg3.fit(df[["R&D Spend",'Administration','Marketing Spend','Florida','New York']],df.Profit)


# In[21]:


df


# In[22]:


a48=reg3.predict([[542,51743,0,0,10]])
Y=df['Profit']
Y


# In[23]:


y=df.iloc[:,4].values
y


# In[ ]:




