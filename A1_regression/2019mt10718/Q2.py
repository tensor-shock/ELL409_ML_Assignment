#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[4]:


train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')


# In[5]:


train


# In[6]:


ids=[x.split('/') for x in train['id']]
date=[]
month=[]
year=[]
for x in ids:
    date.append(int(x[1]))
    month.append(int(x[0]))
    year.append(int(x[2]))
train['date']=date
train['month']=month
train['year']=year


# In[7]:


ids=[x.split('/') for x in test['id']]
date=[]
month=[]
year=[]
for x in ids:
    date.append(int(x[1]))
    month.append(int(x[0]))
    year.append(int(x[2]))
test['date']=date
test['month']=month
test['year']=year


# In[8]:


train


# In[9]:


value=list(train['value'])
month=list(train['month'])
year=list(train['year'])


# In[10]:


test


# In[11]:


train.dtypes


# In[ ]:





# In[ ]:





# In[ ]:





# In[152]:


norm=120#normalize


# In[153]:


train['xfunc']=(12*train['year']+train['month'])/norm
#xfunc=12yr+mo


# In[154]:


train.head()


# In[155]:


xfunc=list(train['xfunc'])


# In[156]:


xmin=min(xfunc)
xmax=max(xfunc)
print(xmin,xmax)


# In[157]:


plt.figure(figsize=(14,6))
sns.lineplot(xfunc,value,marker='o',label='value')
plt.show()


# In[158]:


train_part=train[:100]
valid_part=train[100:]


# In[159]:


xdata=train_part.xfunc.values
tdata=train_part.value.values
xvalid=valid_part.xfunc.values
tvalid=valid_part.value.values


# In[160]:


#TRYING POLYNOMIAL AND FAILING MISERABLY


# In[161]:


#import numpy as np
#xdata=np.genfromtxt('gaussian.csv',usecols=(0),delimiter=',')
#tdata=np.genfromtxt('gaussian.csv',usecols=(1),delimiter=',')

def square_error(m,t_i,w,x_i):
    phi=make_phi(m,x_i)
    w_t=w[:,None].transpose()
    wphi=np.matmul(w_t,phi[:,None])
    f=np.sum(wphi)
    diff=t_i-f
    return diff**2

def sum_square_error(m,n,t,w,x):
    total=0
    for i in range(n):
        total+=square_error(m,t[i],w,x[i])
    return total/2

def rms_error(sse,n):
    return (sse*2/n)**0.5

def make_phi(m,x_i):
    phi=np.zeros(m)
    for i in range(m):
        phi[i]=x_i**i
    return phi

def des_mat(m,n,x):
    des= np.empty((0,m))
    
    for i in range(n):
        phi=make_phi(m,x[i])
        des=np.append(des,[phi],axis=0)
    return des

def reg_sum_square_error(lamb,m,n,t,w,x):
    sse=sum_square_error(m,n,t,w,x)
    w_t=w[:,None].transpose()
    rege=np.sum((lamb/2)*np.matmul(w_t,w[:,None]))
    return rege+sse

m=3
n=4
lamb=0

def least_square_regression_pinv(x,t,m,n):
    x=xdata[:n]
    t=tdata[:n]
    des=des_mat(m,n,x)
    moopeninv=np.linalg.pinv(des)
    wml=np.matmul(moopeninv,t)
    res=sum_square_error(m,n,t,wml,x)
    rmsres=rms_error(res,n)
    return wml,rmsres

def reg_least_square_regression_pinv(xdata,tdata,m,n,lamb):
    x=xdata[:n]
    t=tdata[:n]
    des=des_mat(m,n,x)
    #pinv
    '''
    pinv=np.matmul(des.transpose(),des)
    a=np.identity(m)
    a=a*lamb
    pinv=np.add(pinv,a)
    pinv=np.linalg.inv(pinv)
    pinv=np.matmul(pinv,des.transpose())
    '''
    pinv = np.matmul(np.linalg.pinv(lamb*np.identity(m)+np.matmul(np.transpose(des),des)),np.transpose(des))
    #print('reg',pinv)
    wmlreg=np.matmul(pinv,t)
    res=sum_square_error(m,n,t,wmlreg,x)
    resreg=reg_sum_square_error(lamb,m,n,t,wmlreg,x)
    rmsres=rms_error(res,n)
    rmsresreg=rms_error(resreg,n)
    return wmlreg,rmsresreg,rmsres


# In[162]:


#testing


# In[163]:


errmin=1e43
for i in range(100):
    wml,rmsres=least_square_regression_pinv(xdata,tdata,i,100)
    err=sum_square_error(i,10,tvalid,wml,xvalid)
    testerr=rms_error(err,10)
    print(i,testerr,rmsres)
    if errmin>testerr:
        mmin=i
        errmin=testerr
print(mmin,errmin)


# In[164]:


errmin


# In[165]:


test['xfunc']=(12*test['year']+test['month'])/norm


# In[166]:


test


# In[167]:


xfunctest=list(test.xfunc)


# In[168]:


wml,rmsres=least_square_regression_pinv(xdata,tdata,80,100)


# In[169]:


xtest=test['xfunc'].values


# In[170]:


xtest


# In[171]:


def get_output(m,n,x_test,wml):
    pred=[]
    for i in range(n):
        phi=make_phi(m,x_test[i])
        res=np.matmul(wml.transpose(),phi)
        ans=np.sum(res)
        pred.append(ans)
    return pred


# In[172]:


pred=get_output(80,10,xtest,wml)


# In[173]:


pred


# In[174]:


plt.figure(figsize=(14,6))
sns.lineplot(xfunc,value,marker='o',label='value')
sns.lineplot(xfunctest,pred,marker='o',label='pred')
plt.show()


# In[ ]:





# In[177]:


xp=[i/100 for i in range(49,148)]
#xp=[i/1000 for i in range(236,712)]
yp=[]
for x_coord in xp:
    phi=make_phi(80,x_coord)
    yp.append(np.sum(np.matmul(wml.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xfunc,value,marker='o',label='value')
#sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual data vs predicted regularized polynomial fit')
plt.show()


# In[178]:


#Second approach sinusoidal??


# In[179]:


print(xdata.shape,xvalid.shape)


# In[180]:


test


# In[272]:


#import numpy as np
#xdata=np.genfromtxt('gaussian.csv',usecols=(0),delimiter=',')
#tdata=np.genfromtxt('gaussian.csv',usecols=(1),delimiter=',')

def square_error(m,t_i,w,x_i):
    phi=make_phi(m,x_i)
    w_t=w[:,None].transpose()
    wphi=np.matmul(w_t,phi[:,None])
    f=np.sum(wphi)
    diff=t_i-f
    return diff**2

def sum_square_error(m,n,t,w,x):
    total=0
    for i in range(n):
        total+=square_error(m,t[i],w,x[i])
    return total/2

def rms_error(sse,n):
    return (sse*2/n)**0.5

def make_phi(m,x_i):
    phi=np.zeros(m)
    for i in range(m):
        phi[i]=np.sin(np.pi*x_i)**i+2
        #phi[i]=np.sin(x_i)**(i)+1
    return phi

def des_mat(m,n,x):
    des= np.empty((0,m))
    
    for i in range(n):
        phi=make_phi(m,x[i])
        des=np.append(des,[phi],axis=0)
    return des

def reg_sum_square_error(lamb,m,n,t,w,x):
    sse=sum_square_error(m,n,t,w,x)
    w_t=w[:,None].transpose()
    rege=np.sum((lamb/2)*np.matmul(w_t,w[:,None]))
    return rege+sse


def least_square_regression_pinv(x,t,m,n):
    x=xdata[:n]
    t=tdata[:n]
    des=des_mat(m,n,x)
    moopeninv=np.linalg.pinv(des)
    wml=np.matmul(moopeninv,t)
    res=sum_square_error(m,n,t,wml,x)
    rmsres=rms_error(res,n)
    return wml,rmsres

def reg_least_square_regression_pinv(xdata,tdata,m,n,lamb):
    x=xdata[:n]
    t=tdata[:n]
    des=des_mat(m,n,x)
    #pinv
    '''
    pinv=np.matmul(des.transpose(),des)
    a=np.identity(m)
    a=a*lamb
    pinv=np.add(pinv,a)
    pinv=np.linalg.inv(pinv)
    pinv=np.matmul(pinv,des.transpose())
    '''
    pinv = np.matmul(np.linalg.pinv(lamb*np.identity(m)+np.matmul(np.transpose(des),des)),np.transpose(des))
    #print('reg',pinv)
    wmlreg=np.matmul(pinv,t)
    res=sum_square_error(m,n,t,wmlreg,x)
    resreg=reg_sum_square_error(lamb,m,n,t,wmlreg,x)
    rmsres=rms_error(res,n)
    rmsresreg=rms_error(resreg,n)
    return wmlreg,rmsresreg,rmsres


# In[273]:


train_part=train[:100]
valid_part=train[100:]


# In[274]:


xdata=train_part.xfunc.values
tdata=train_part.value.values
xvalid=valid_part.xfunc.values
tvalid=valid_part.value.values


# In[275]:


errmin=1e43
for i in range(500):
    wml,rmsres=least_square_regression_pinv(xdata,tdata,i,100)
    err=sum_square_error(i,10,tvalid,wml,xvalid)
    testerr=rms_error(err,10)
    print(i,testerr,rmsres)
    if errmin>testerr:
        mmin=i
        errmin=testerr
print(mmin,errmin)


# In[276]:


print(mmin,errmin)


# In[277]:


#mmin=53


# In[278]:


wml,rmsres=least_square_regression_pinv(xdata,tdata,mmin,100)


# In[279]:


pred=get_output(mmin,10,xtest,wml)


# In[280]:


plt.figure(figsize=(14,6))
sns.lineplot(xfunc,value,marker='o',label='value')
sns.lineplot(xfunctest,pred,marker='o',label='pred')
plt.show()


# In[281]:


xp=[i/100 for i in range(49,148)]
yp=[]
for x_coord in xp:
    phi=make_phi(mmin,x_coord)
    yp.append(np.sum(np.matmul(wml.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xfunc,value,marker='o',label='value')
#sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual data vs predicted regularized polynomial fit')
plt.show()


# In[ ]:





# In[495]:


# sin(i*x_i)             2.3461
# sin(x_i**i)            9
# sin(x_i)**i+1          2.84
# sin(x_i)**i+x_i        2.86
# sin(np.pi*x_i)**i      1.917697855475552 current best
# sin(np.pi*x_i)**i+1    1.9174673875893822
# sin(np.pi*x_i)**i+2    1.9173773875893822
# sin(np.pi*x_i)**i+i    1.918
# sin(np.pi*x_i)**i+x_i  1.9172379881187527 doesnt beat best


# In[271]:


#best normalizer bcos sin function #failed
train_part=train[:100]
valid_part=train[100:]
erransmin=1e22
for j in range(1,500):
    xdata=train_part.xfunc.values/j
    tdata=train_part.value.values/j
    xvalid=valid_part.xfunc.values/j
    tvalid=valid_part.value.values/j
    
    errmin=1e43
    for i in range(500):
        wml,rmsres=least_square_regression_pinv(xdata,tdata,i,100)
        err=sum_square_error(i,10,tvalid,wml,xvalid)
        testerr=rms_error(err,10)
        #print(i,testerr,rmsres)
        if errmin>testerr:
            mmin=i
            errmin=testerr
    print(j,mmin,errmin)
    if errmin<erransmin:
        mini=j
        erransmin=errmin
print(mini,erransmin)


# In[282]:


output=pd.DataFrame(test['id'])
output['value']=pred


# In[283]:


output


# In[284]:


output.to_csv('output.csv',index=False)

