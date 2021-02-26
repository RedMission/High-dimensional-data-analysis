#!/usr/bin/env python
# coding: utf-8

# In[2]:


def t(name):
    b = "get_ipython().system('jupyter nbconvert --to python"+name+" .ipynb')"
    try:    
        b
    # python即转化为.py，script即转化为.html
    # file_name.ipynb即当前module的文件名
    except:
        pass


# In[ ]:




