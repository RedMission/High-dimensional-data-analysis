#!/usr/bin/env python
# coding: utf-8

# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis

pd.set_option('display.max_columns', None)
#cities_10记录了十个沿海省份的经济指标，通过降维方法对各省、市经济发展情况评估
'''变量 	含义
AREA 	地区
X1 	GDP
X2 	人均GDP
X3 	工业增加值
X4 	第三产业增加值
X5 	固定资产投资
X6 	基本建设投资
X7 	社会消费品零售总额
X8 	海关出口总额
X9 	地方财政收入'''


# In[15]:


#导入数据
cities = pd.read_csv("cities_10.csv", encoding='gbk')
cities


# In[16]:


cities.loc[:, 'X1':].corr(method='pearson').head()


# In[17]:


#主成分分析（预先进行标准化）
from sklearn.preprocessing import scale, StandardScaler
n_components = 2

ss = StandardScaler()
scale_cities = ss.fit_transform(cities.ix[:, 'X1':])

pca = PCA(n_components=n_components)
pca.fit(scale_cities)
pca.explained_variance_ratio_


# In[18]:


pd.DataFrame(pca.components_)


# In[19]:


pca_scores = pd.DataFrame(pca.transform(scale_cities), columns=['Gross', 'Avg'])
scale_pca_cities = cities.join(pca_scores)
scale_pca_cities


# In[20]:


from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体

x = scale_pca_cities['Gross']
y = scale_pca_cities['Avg']
label = scale_pca_cities['AREA']

plt.figure(figsize=(8,5))
plt.grid(linestyle='-.')
plt.scatter(x, y)
plt.xlabel('Gross')
plt.ylabel('Avg')
for a, b, l in zip(x, y, label):
    plt.text(a, b+0.1, '%s' % l, ha='center', va= 'bottom',fontsize=13)

plt.show()


# In[21]:


#稀疏主成分分析 SparsePCA
alpha = 0.5
spca = SparsePCA(n_components=n_components, alpha=alpha)
spca.fit(scale_cities)
pd.DataFrame(spca.components_)


# In[22]:


#选择合适的稀疏度
from functools import reduce

n_cols = scale_cities.shape[1]
spca_with_alpha = []

for i in np.arange(0.1, 1, 0.05):
    spca = SparsePCA(n_components=n_components, alpha=i)
    spca.fit(scale_cities)
    components_ = pd.DataFrame(spca.components_).T
    diff = np.count_nonzero(components_) - n_cols
    spca_with_alpha.append((i, diff))

selector = pd.DataFrame(spca_with_alpha)
selector.T


# In[23]:


spca_ = SparsePCA(n_components=n_components, alpha=0.65).fit(scale_cities)
pd.DataFrame(spca_.components_)


# In[24]:


scale_spca_cities = pd.DataFrame(spca_.transform(scale_cities), columns=['Gross', 'Avg'])
scale_spca_cities


# In[25]:


x = scale_spca_cities['Gross']
y = scale_spca_cities['Avg']
label = cities['AREA']

plt.figure(figsize=(8,5))
plt.grid(linestyle='-.')
plt.xlabel('Gross')
plt.ylabel('Avg')
plt.scatter(x, y, c='r',marker='^')
for a, b, l in zip(x, y, label):
    plt.text(a, b+0.1, '%s' % l, ha='center', va='bottom',fontsize=14)

plt.show()


# In[ ]:


#把ipynb转为py
try:    
    get_ipython().system('jupyter nbconvert --to python SPCA_example.ipynb')
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
except:
    pass


# In[ ]:




