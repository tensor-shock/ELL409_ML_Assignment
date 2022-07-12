#!/usr/bin/env python
# coding: utf-8

# In[428]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[429]:


#file path
file='gaussian.csv'


# In[430]:


#data taken from file
xdata=np.genfromtxt(file,usecols=(0),delimiter=',')
tdata=np.genfromtxt(file,usecols=(1),delimiter=',')


# In[431]:


#vector of size m making phi(x)
def make_phi(m,x_i):
    phi=np.zeros(m)
    for i in range(m):
        phi[i]=x_i**i
    return phi


# In[432]:


#makes design matrix nXm
def des_mat(m,n,x):
    des= np.empty((0,m))
    for i in range(n):
        phi=make_phi(m,x[i])
        des=np.append(des,[phi],axis=0)
    return des


# In[433]:


#returns (t-y)^2 for one input point(single n value)
def square_error(m,t_i,w,x_i):
    phi=make_phi(m,x_i)
    w_t=w[:,None].transpose()
    wphi=np.matmul(w_t,phi[:,None])
    y=np.sum(wphi)
    diff=t_i-y
    return diff**2#mean square error(L2)
    #return abs(diff)#L1 error#min test error is lesser than L2
    #return np.exp(abs(diff))


# In[434]:


#SSE for n data points
def sum_square_error(m,n,t,w,x):
    total=0
    for i in range(n):
        total+=square_error(m,t[i],w,x[i])
    return total/2


# In[435]:


#RMS error given SSE for n points
def rms_error(sse,n):
    return (sse*2/n)**0.5


# In[436]:


#complete moore penrose pseudo inverse
def least_square_regression_pinv(x,t,m,n):
    des=des_mat(m,n,x)
    moopeninv=np.linalg.pinv(des)
    wml=np.matmul(moopeninv,t)
    res=sum_square_error(m,n,t,wml,x)
    rmsres=rms_error(res,n)
    return wml,res,rmsres


# In[438]:


xmin=(min(xdata))
xmax=(max(xdata))
print(xmin)
print(xmax)


# In[472]:


#error vs train size
errs=[]
for i in range(10,100,10):
    train_size=i
    test_size=100-train_size
    xtrain=xdata[:train_size]
    ttrain=tdata[:train_size]
    xvalid=xdata[train_size:]
    tvalid=tdata[train_size:]
    m_range=30
    errmin=1e33 #stores minimum error
    training_rms=[]
    testing_rms=[]
    for i in range(m_range):
        m=i
        n=train_size
        wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
        sse=sum_square_error(m,test_size,tvalid,wml,xvalid)
        rms=rms_error(sse,test_size)
        #print(i,rmsres,rms)
        training_rms.append(rmsres)
        testing_rms.append(rms)
        if rms<errmin:
            errmin=rms
            wml_best=wml
            m_best=m
    #print('train_size: ',i ,'m: ',m_best,'\nerr: ',errmin,'\nwml: ',wml_best)
    errs.append(errmin)
xp=[i for i in range(10,100,10)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,errs,marker='o',label='minimum rms error')
plt.title('min RMS_error vs train_size value')
plt.grid(True)
plt.show()


# In[473]:


#error vs degree of poly
#80 20 train test split
#min test error found at 80 20 split
train_size=80
test_size=100-train_size
xtrain=xdata[:train_size]
ttrain=tdata[:train_size]
xvalid=xdata[train_size:]
tvalid=tdata[train_size:]
m_range=30
errmin=1e33
training_rms=[]
testing_rms=[]
for i in range(m_range):
    m=i
    n=train_size
    wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
    sse=sum_square_error(m,test_size,tvalid,wml,xvalid)
    rms=rms_error(sse,test_size)
    print(i,rmsres,rms)
    training_rms.append(rmsres)
    testing_rms.append(rms)
    if rms<errmin:
        errmin=rms
        wml_best=wml
        m_best=m
print('m optimal: ',m_best,'\nerr optimal: ',errmin,'\nwml optimal: ',wml_best)
xp=[i for i in range(m_range)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,training_rms,marker='o',label='training rms error')
sns.lineplot(xp,testing_rms,marker='o',label='testing rms error')
plt.title('RMS_error vs m value')
plt.show()


# In[441]:


#best fit poly


#xp=[i/100 for i in range(-243,341)]#non gaussian fit
xp=[i/100 for i in range(-140,256)]#gaussian fit
yp=[]
for x_coord in xp:
    phi=make_phi(m_best,x_coord)
    yp.append(np.sum(np.matmul(wml_best.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual dat vs predicted polynomial')
plt.show()


# In[474]:


#optimizing cross val batch number
err=[]
for k in range(2,8):
    data_points_total=100
    val_set_size=int(data_points_total/k)
    train_set_size=data_points_total-val_set_size
    m_avg=[]
    testing_errors=[]
    for val_step in range(k):
        test_begin_index=k*val_step
        test_end_index=min(test_begin_index+val_set_size,data_points_total)
        xtrain=np.append(xdata[:test_begin_index],xdata[test_end_index:])
        ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
        xvalid=xdata[test_begin_index:test_end_index]
        tvalid=tdata[test_begin_index:test_end_index]
        m_range=30
        errmin=1e33
        for i in range(m_range):
            m=i
            n=train_set_size
            wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
            sse=sum_square_error(m,val_set_size,tvalid,wml,xvalid)
            rms=rms_error(sse,val_set_size)
            if rms<errmin:
                errmin=rms
                wml_best=wml
                m_best=m
        m_avg.append(m_best)
        testing_errors.append(errmin)
    #print('m of rounds: ',m_avg)
    #print('testing errors of rounds: ',testing_errors)
    m_final=int(np.sum(m_avg)/k)
    #print('m final',m_final)
    wml,res,rmsres=least_square_regression_pinv(xdata,tdata,m_final,data_points_total)
    #print('rms error for m_final',rmsres)
    err.append(rmsres)
#best fit poly after cross_validation
xp=[i for i in range(2,8)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,err,marker='o',label='rms error')
plt.title('rms error vs number of cross-validation batches')
plt.grid(True)
plt.show()


# In[458]:


#k batch cross val (k taken to be 6 gives good result)
data_points_total=100
k=6
val_set_size=int(data_points_total/k)
train_set_size=data_points_total-val_set_size
m_avg=[]
testing_errors=[]
for val_step in range(k):
    test_begin_index=k*val_step
    test_end_index=min(test_begin_index+val_set_size,data_points_total)
    xtrain=np.append(xdata[:test_begin_index],xdata[test_end_index:])
    ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
    xvalid=xdata[test_begin_index:test_end_index]
    tvalid=tdata[test_begin_index:test_end_index]
    m_range=30
    errmin=1e33
    for i in range(m_range):
        m=i
        n=train_set_size
        wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
        sse=sum_square_error(m,val_set_size,tvalid,wml,xvalid)
        rms=rms_error(sse,val_set_size)
        if rms<errmin:
            errmin=rms
            wml_best=wml
            m_best=m
    m_avg.append(m_best)
    testing_errors.append(errmin)
print('m of rounds: ',m_avg)
print('testing errors of rounds: ',testing_errors)
m_final=int(np.sum(m_avg)/k)
print('m final',m_final)
wml,res,rmsres=least_square_regression_pinv(xdata,tdata,m_final,data_points_total)
#print('rms error for m_final',rmsres)
print('error for m_final',rmsres)
#best fit poly after cross_validation
#xp=[i/100 for i in range(-243,341)]
xp=[i/100 for i in range(-140,256)]
yp=[]
for x_coord in xp:
    phi=make_phi(m_final,x_coord)
    yp.append(np.sum(np.matmul(wml.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual data vs predicted polynomial fit')
plt.show()


# In[476]:


#noise variantion
pred=[]

for i in range(100):
    phi=make_phi(m_final,xdata[i])
    pred.append(np.sum(np.matmul(wml.transpose(),phi)))
'''
m_final=8
wml,res,rmsres=least_square_regression_pinv(xdata,tdata,m_final,20)
for i in range(20):
    phi=make_phi(m_final,xdata[i])
    pred.append(np.sum(np.matmul(wml.transpose(),phi)))
'''
#noise=np.absolute(np.subtract(tdata,pred))
noise=(np.subtract(tdata,pred))
plt.figure(figsize=(14,6))
sns.lineplot(xdata,pred,marker='o',label='predicted data')
sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual vs predicted data')
plt.show()
plt.figure(figsize=(14,6))
sns.lineplot(xdata,noise,marker='o',label='noise')
plt.title('noise variation')
plt.show()
print('mean of noise',np.mean(noise))
print('variance of noise',np.var(noise))


# In[445]:


noise.shape


# In[477]:


#noise estimation
#80 20 train test split
#min test error found at 80 20 split
train_size=80
test_size=100-train_size
xtrain=xdata[:train_size]
ttrain=noise[:train_size]
xvalid=xdata[train_size:]
tvalid=noise[train_size:]
m_range=30
errmin=1e33
training_rms=[]
testing_rms=[]
for i in range(m_range):
    m=i
    n=train_size
    wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
    sse=sum_square_error(m,test_size,tvalid,wml,xvalid)
    rms=rms_error(sse,test_size)
    print(i,rmsres,rms)
    training_rms.append(rmsres)
    testing_rms.append(rms)
    if rms<errmin:
        errmin=rms
        wml_best=wml
        m_best=m
print('m optimal: ',m_best,'\nerr optimal: ',errmin,'\nwml optimal: ',wml_best)
xp=[i for i in range(m_range)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,training_rms,marker='o',label='training rms error')
sns.lineplot(xp,testing_rms,marker='o',label='testing rms error')
plt.title('RMS_error vs m value')
plt.show()


# In[461]:



#xp=[i/100 for i in range(-243,341)]
xp=[i/100 for i in range(-140,256)]
yp=[]
for x_coord in xp:
    phi=make_phi(m_best,x_coord)
    yp.append(np.sum(np.matmul(wml_best.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xdata,noise,marker='o',label='actual data')
plt.title('actual dat vs predicted polynomial')
plt.show()


# In[478]:


#poly fit of noise
m=10
wml,res,rmsres=least_square_regression_pinv(xtrain,ttrain,m,n)
xp=[i/100 for i in range(-140,256)]
#xp=[i/100 for i in range(-243,341)]
yp=[]
for x_coord in xp:
    phi=make_phi(m,x_coord)
    yp.append(np.sum(np.matmul(wml.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial noise')
sns.lineplot(xdata,noise,marker='o',label='actual noise')
plt.title('actual dat vs predicted noise')
plt.show()


# In[449]:


#regularized


# In[479]:


def reg_sum_square_error(lamb,m,n,t,w,x):
    sse=sum_square_error(m,n,t,w,x)
    rege=np.sum((lamb/2)*np.matmul(w.transpose(),w))
    return rege+sse


# In[480]:


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
    #res=sum_square_error(m,n,t,wmlreg,x)
    resreg=reg_sum_square_error(lamb,m,n,t,wmlreg,x)
    #rmsres=rms_error(res,n)
    rmsresreg=rms_error(resreg,n)
    return wmlreg,resreg,rmsresreg


# In[481]:


wmlreg,resreg,rmsresreg=reg_least_square_regression_pinv(xdata,tdata,26,100,0)
print(wmlreg)


# In[482]:


#redundant exploratory code block
train_size=80
test_size=100-train_size
xtrain=xdata[:train_size]
ttrain=tdata[:train_size]
xvalid=xdata[train_size:]
tvalid=tdata[train_size:]
m_range=20
errmin=1e33
lamb=1
training_rms=[]
testing_rms=[]
for i in range(m_range):
    m=i
    n=train_size
    wmlreg,resreg,rmsresreg=reg_least_square_regression_pinv(xtrain,ttrain,m,n,lamb)
    sse=reg_sum_square_error(lamb,m,test_size,tvalid,wmlreg,xvalid)
    rms=rms_error(sse,test_size)
    print(i,rmsresreg,resreg)
    training_rms.append(rmsresreg)
    testing_rms.append(rms)
    if rms<errmin:
        errmin=rms
        wml_best=wmlreg
        m_best=m
print('m: ',m_best,'\nerr: ',errmin,'\nwml: ',wml_best)

xp=[i for i in range(m_range)]
plt.figure(figsize=(14,6))
sns.lineplot(xp,training_rms,marker='o',label='training rms error')
sns.lineplot(xp,testing_rms,marker='o',label='testing rms error')
plt.title('RMS_error vs m value')
plt.show()


# In[483]:


#error vs lambda
lambdas=[i for i in range(-15,0)]
train_size=19
test_size=20-train_size
xtrain=xdata[:train_size]
ttrain=tdata[:train_size]
xvalid=xdata[train_size:]
tvalid=tdata[train_size:]
training_rms=[]
testing_rms=[]
for lambd in lambdas:
    m=m_final
    n=train_size
    lamb=np.exp(lambd)
    wmlreg,resreg,rmsresreg=reg_least_square_regression_pinv(xtrain,ttrain,m,n,lamb)
    sse=reg_sum_square_error(lamb,m,test_size,tvalid,wmlreg,xvalid)
    rms=rms_error(sse,test_size)
    print(lamb,rmsresreg,rms)
    training_rms.append(rmsresreg)
    testing_rms.append(rms)
plt.figure(figsize=(14,6))
#sns.lineplot(lambdas,training_rms,marker='o',label='training rms error')
sns.lineplot(lambdas,testing_rms,marker='o',label='testing rms error')
plt.title('RMS_error vs ln(lamb) value ,testing')
plt.grid(True)
plt.show()
plt.figure(figsize=(14,6))
sns.lineplot(lambdas,training_rms,marker='o',label='training rms error')
#sns.lineplot(lambdas,testing_rms,marker='o',label='testing rms error')
plt.title('RMS_error vs ln(lamb) value, training')
plt.grid(True)
plt.show()


# In[484]:


#k batch cross val (k taken to be 6 gives good result)
data_points_total=100
k=6
lamb=np.exp(-11)
val_set_size=int(data_points_total/k)
train_set_size=data_points_total-val_set_size
testing_errors=[]
m_avg=[]
for val_step in range(k):
    test_begin_index=k*val_step
    test_end_index=min(test_begin_index+val_set_size,data_points_total)
    xtrain=np.append(xdata[:test_begin_index],xdata[test_end_index:])
    ttrain=np.append(tdata[:test_begin_index],tdata[test_end_index:])
    xvalid=xdata[test_begin_index:test_end_index]
    tvalid=tdata[test_begin_index:test_end_index]
    m_range=30
    errmin=1e33
    for i in range(m_range):
        m=i
        n=train_set_size
        wml,res,rmsres=reg_least_square_regression_pinv(xtrain,ttrain,m,n,lamb)
        sse=reg_sum_square_error(lamb,m,val_set_size,tvalid,wml,xvalid)
        rms=rms_error(sse,val_set_size)
        if rms<errmin:
            errmin=rms
            #wml_best=wml
            m_best=m
    m_avg.append(m_best)
    testing_errors.append(errmin)
print('m of rounds: ',m_avg)
print('testing errors of rounds: ',testing_errors)
m_final=int(np.sum(m_avg)/k)
print('m final',m_final)
wml,res,rmsres=reg_least_square_regression_pinv(xdata,tdata,m_final,data_points_total,lamb)
print('rms error for m_final',rmsres)


# In[485]:


#best fit poly after cross_validation
xp=[i/100 for i in range(-140,256)]
yp=[]
for x_coord in xp:
    phi=make_phi(m_final,x_coord)
    yp.append(np.sum(np.matmul(wml.transpose(),phi)))
plt.figure(figsize=(14,6))
sns.lineplot(xp,yp,marker='o',label='predicted polynomial function')
sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual data vs predicted regularized polynomial fit')
plt.show()


# In[486]:


#noise variance
pred=[]
for i in range(100):
    phi=make_phi(m_final,xdata[i])
    pred.append(np.sum(np.matmul(wml.transpose(),phi)))
#noise=np.absolute(np.subtract(tdata,pred))
noise=(np.subtract(tdata,pred))
plt.figure(figsize=(14,6))
sns.lineplot(xdata,pred,marker='o',label='predicted data')
sns.lineplot(xdata,tdata,marker='o',label='actual data')
plt.title('actual vs predicted data')
plt.show()
plt.figure(figsize=(14,6))
sns.lineplot(xdata,noise,marker='o',label='noise')
plt.title('noise variation')
plt.show()
print('mean of noise',np.mean(noise))
print('variance of noise',np.var(noise))

