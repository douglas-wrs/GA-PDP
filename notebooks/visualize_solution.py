#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


# In[2]:


df = pd.read_pickle("df_individual.pkl")


# In[3]:


df_route = pd.read_pickle("df_route.pkl")


# In[4]:


min_longitude = -49.3355
max_longitude = -49.2067
min_latitude = -16.6434
max_latitude = -16.7516

BBox = ((min_longitude, max_longitude, min_latitude, max_latitude))


# In[5]:


gyn_m = plt.imread('map.png')


# In[6]:


len(df)


# In[7]:


fig, ax = plt.subplots(figsize = (10,10))

individual_longitude = df.longitude
individual_latitude = df.latitude
# ax.scatter(individual_longitude, individual_latitude, zorder=1, alpha= 1, c='r', s=30)

ax.set_title('Indivual Solution')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])

request_longitude = df_route[[2]].to_numpy()
request_latitude = df_route[[3]].to_numpy()

ax.scatter(request_longitude, request_latitude, zorder=1, alpha= 1, c='b', s=30)

for i in range(0, len(df), 1):
    plt.plot([request_longitude[i], individual_longitude[i]], [request_latitude[i], individual_latitude[i]], 'k-', alpha= 1)
       
ax.imshow(gyn_m, alpha= 0.6, zorder=0, extent = BBox, aspect= 'equal')

ax.axis('off')
plt.show()

