#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
from libsvm.svm import *
from libsvm.svmutil import *
import matplotlib.pyplot as plt


# In[36]:


import seaborn as sns
import time


# In[37]:


#file path
file='2019MT10718.csv'


# In[38]:


#data taken from file
data=np.genfromtxt(file,delimiter=',')
#req_data=[x for x in data if x[25] in [2,9]]
#data=np.array(req_data)


# In[39]:


data.shape


# In[40]:


data[:5]


# In[55]:


xdata=data[:,:25]
tdata=data[:,25:].flatten()
#xdata=data[:,:10]

#tdata=tdata.reshape(tdata.shape[0],1)


# In[56]:


tdata.shape


# In[57]:


tdata.dtype


# In[58]:


model = svm_train(tdata, xdata)

start_time = time.time()
p_labs, p_acc, p_vals = svm_predict(tdata, xdata, model )
print("--- %s seconds ---" % (time.time() - start_time))


# In[59]:


err=[]
min_lim=2
max_lim=21
for k in range(min_lim,max_lim):
    #k batch cross val (k taken to be 6 gives good result)
    data_points_total=xdata.shape[0]
    val_set_size=int(data_points_total/k)
    train_set_size=data_points_total-val_set_size
    accuracy=[]
    print(k)
    for val_step in range(k):
        test_begin_index=k*val_step
        test_end_index=min(test_begin_index+val_set_size,data_points_total)
        xtrain=np.append(xdata[:test_begin_index][:],xdata[test_end_index:][:])
        ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
        r=ttrain.shape[0]
        c=int(xtrain.shape[0]/r)
        xtrain=xtrain.reshape(r,c)
        xvalid=xdata[test_begin_index:test_end_index]
        tvalid=tdata[test_begin_index:test_end_index]
        #print(xvalid.shape,xtrain.shape)
        #print(tvalid)
        model = svm_train(ttrain, xtrain)
        p_labs, p_acc, p_vals = svm_predict(tvalid, xvalid, model )
        accuracy.append(p_acc[0])
    print(np.mean(accuracy))
    err.append(np.mean(accuracy))
#best fit poly after cross_validation
xp=[i for i in range(min_lim,max_lim)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,err,marker='o',label='accuracy')
plt.title('accuracy vs number of cross-validation batches')
plt.grid(True)
plt.show()


# In[64]:


#12 batch cross val 
data_points_total=xdata.shape[0]
k=15
val_set_size=int(data_points_total/k)
train_set_size=data_points_total-val_set_size
accuracy=[]
for val_step in range(k):
    test_begin_index=k*val_step
    test_end_index=min(test_begin_index+val_set_size,data_points_total)
    xtrain=np.append(xdata[:test_begin_index][:],xdata[test_end_index:][:])
    ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
    r=ttrain.shape[0]
    c=int(xtrain.shape[0]/r)
    xtrain=xtrain.reshape(r,c)
    xvalid=xdata[test_begin_index:test_end_index]
    tvalid=tdata[test_begin_index:test_end_index]
    #print(xvalid.shape,xtrain.shape)
    #print(tvalid)
    model = svm_train(ttrain, xtrain,'-t 2'  )
    start_time = time.time()
    p_labs, p_acc, p_vals = svm_predict(tvalid, xvalid, model )
    print("--- %s seconds ---" % (time.time() - start_time))
    accuracy.append(p_acc[0])

print('Number of batches',k)
print('Number of features are', 10)
#print('gamma value is',2)
print('final accuracy',np.mean(accuracy))


# In[61]:


#15 batch cross val 
data_points_total=xdata.shape[0]
k=12
val_set_size=int(data_points_total/k)
train_set_size=data_points_total-val_set_size
accuracy=[]
vals=[i for i in range(0,30)]
acc_val=[]
for i in vals:
    for val_step in range(k):
        test_begin_index=k*val_step
        test_end_index=min(test_begin_index+val_set_size,data_points_total)
        xtrain=np.append(xdata[:test_begin_index][:],xdata[test_end_index:][:])
        ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
        r=ttrain.shape[0]
        c=int(xtrain.shape[0]/r)
        xtrain=xtrain.reshape(r,c)
        xvalid=xdata[test_begin_index:test_end_index]
        tvalid=tdata[test_begin_index:test_end_index]
        #print(xvalid.shape,xtrain.shape)
        #print(tvalid)
        #pars='-t 3 -g '+str(i)
        print(pars)
        model = svm_train(ttrain, xtrain ,pars )
        start_time = time.time()
        p_labs, p_acc, p_vals = svm_predict(tvalid, xvalid, model )
        print("--- %s seconds ---" % (time.time() - start_time))
        accuracy.append(p_acc[0])

    print('Number of batches',k)
    #print('Number of samples are', 10)
    #print('gamma value is',2)
    print('final accuracy',np.mean(accuracy))
    acc_val.append(np.mean(accuracy))
plt.figure(figsize=(14,6))
sns.lineplot(vals,acc_val,marker='o',label='accuracy')
plt.title('accuracy vs value of gamma')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




