#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[14]:


df_corner = pd.read_pickle("df_result_corner.pkl")
df_corner['METHOD'] = 'CORNER'
df_door = pd.read_pickle("df_result_door.pkl")
df_ga0 = pd.read_pickle("df_result_ga0.pkl")
df_ga1 = pd.read_pickle("df_result_ga1.pkl")
df_ga2 = pd.read_pickle("df_result_ga2.pkl")


# In[15]:


df_all = pd.concat([df_corner, df_door, df_ga0, df_ga1, df_ga2], sort=False)


# In[19]:


df_all.columns


# In[23]:


df_all.boxplot(by="METHOD")

