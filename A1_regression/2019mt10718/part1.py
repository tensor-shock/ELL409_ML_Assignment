import argparse  
import numpy as np


def setup():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--method", default="pinv", help = "type of solver")  
    parser.add_argument("--batch_size", default=5, type=int, help = "batch size")
    parser.add_argument("--lamb", default=0, type=float, help = "regularization constant")
    parser.add_argument("--polynomial", default=10, type=int, help = "degree of polynomial")
    parser.add_argument("--result_dir", default="", type=str, help = "Files to store plots")  
    parser.add_argument("--X", default="gaussian.csv", type=str, help = "Read content from the file")
    return parser.parse_args()
    
if __name__ == '__main__':
    args = setup()

filepath=args.X
method=args.method
lamb=args.lamb
polynomial=args.polynomial
batch_size=args.batch_size

xdata=np.genfromtxt(filepath,usecols=(0),delimiter=',')
tdata=np.genfromtxt(filepath,usecols=(1),delimiter=',')
data_size=xdata.shape[0]

#vector of size m making phi(x)
def make_phi(m,x_i):
    phi=np.zeros(m)
    for i in range(m):
        phi[i]=x_i**i
    return phi

#makes design matrix nXm
def des_mat(m,n,x):
    des= np.empty((0,m))
    for i in range(n):
        phi=make_phi(m,x[i])
        des=np.append(des,[phi],axis=0)
    return des

#returns (t-y)^2 for one input point(single n value)
def square_error(m,t_i,w,x_i):
    phi=make_phi(m,x_i)
    w_t=w[:,None].transpose()
    wphi=np.matmul(w_t,phi[:,None])
    y=np.sum(wphi)
    diff=t_i-y
    return diff**2#mean square error(L2)
    #return abs(diff)#L1 error
    #return np.exp(abs(diff))

#SSE for n data points
def sum_square_error(m,n,t,w,x):
    total=0
    for i in range(n):
        total+=square_error(m,t[i],w,x[i])
    return total/2

#RMS error given SSE for n points
def rms_error(sse,n):
    return (sse*2/n)**0.5

#complete moore penrose pseudo inverse
def least_square_regression_pinv(x,t,m,n):
    des=des_mat(m,n,x)
    moopeninv=np.linalg.pinv(des)
    wml=np.matmul(moopeninv,t)
    res=sum_square_error(m,n,t,wml,x)
    rmsres=rms_error(res,n)
    return wml,res,rmsres

#regularized error
def reg_sum_square_error(lamb,m,n,t,w,x):
    sse=sum_square_error(m,n,t,w,x)
    rege=np.sum((lamb/2)*np.matmul(w.transpose(),w))
    return rege+sse

#regularized moore penrose 
def reg_least_square_regression_pinv(xdata,tdata,m,n,lamb):
    x=xdata[:n]
    t=tdata[:n]
    des=des_mat(m,n,x)
    #pinv
    pinv = np.matmul(np.linalg.pinv(lamb*np.identity(m)+np.matmul(np.transpose(des),des)),np.transpose(des))
    wmlreg=np.matmul(pinv,t)
    resreg=reg_sum_square_error(lamb,m,n,t,wmlreg,x)
    rmsresreg=rms_error(resreg,n)
    return wmlreg,resreg,rmsresreg

if method=='pinv':
    if lamb==0:
        wml,res,rmsres=least_square_regression_pinv(xdata,tdata,polynomial+1,data_size)
        print('weights=',wml)
    else :
        wmlreg,resreg,rmsresreg=reg_least_square_regression_pinv(xdata,tdata,polynomial+1,data_size,lamb)
        print('weights=',wmlreg)

#gd
        
        
#single data point gradient of error
def der_k_m(m,w,t_k,x_k):
    phi=make_phi(m,x_k)
    wphi=np.matmul(w.transpose(),phi)
    f=np.sum(wphi)
    diff=t_k-f
    phi=diff*phi
    return phi
    
#single data point gradient of error regularized case
def der_k_m_reg(lamb,m,w,t_k,x_k):
    phi=make_phi(m,x_k)
    wphi=np.matmul(w.transpose(),phi)
    f=np.sum(wphi)
    diff=t_k-f
    phi=diff*phi
    phi=np.add(phi,lamb*w)
    return phi
    
#gradient of error summed over entire batch size
def der_err(n,m,t,x,w,lamb=-1):
    dE=np.zeros(m)
    for k in range(n):
        if lamb!=-1:
            ele_k=der_k_m_reg(lamb,m,w,t[k],x[k])
        else:
            ele_k=der_k_m(m,w,t[k],x[k])
        dE=np.add(dE,ele_k)
    return dE

#gradient descent with variable batch sizes
def gd_batch(n,m,x_full,t_full,batch_size=-1,learning_rate=1,epochs=250000):
    w=np.zeros(m)
    err_bound_up=sum_square_error(m,n,t_full,w,x_full)
    if(batch_size==-1):
        batch_size=n
    batch_size=int(batch_size)
    batch_count=int(n/batch_size)
    epochs=int(epochs/batch_count)#modified epochs to control iterations
    for e in range(epochs):
        for b in range(batch_count):
            start_index=b*batch_size
            end_index=min(start_index+batch_size,n)
            x=x_full[start_index:end_index]
            t=t_full[start_index:end_index]
            
            w2=np.add(w,learning_rate*der_err(end_index-start_index,m,t,x,w))
            a=(sum_square_error(m,n,t_full,w2,x_full))
            
            while(a>2*err_bound_up):
                learning_rate/=10
                w2=np.add(w,learning_rate*der_err(end_index-start_index,m,t,x,w))
                a=(sum_square_error(m,n,t_full,w2,x_full))
            
            w=w2
    return a,learning_rate,w

#gradient descent with variable batch sizes regularized
def gd_batch_reg(lamb,n,m,x_full,t_full,batch_size=-1,learning_rate=1,epochs=250000):
    w=np.zeros(m)
    err_bound_up=reg_sum_square_error(lamb,m,n,t_full,w,x_full)
    if(batch_size==-1):
        batch_size=n
    batch_size=int(batch_size)
    batch_count=int(n/batch_size)
    epochs=int(epochs/batch_count)#modified epochs to control iterations
    for e in range(epochs):
        for b in range(batch_count):
            start_index=b*batch_size
            end_index=min(start_index+batch_size,n)
            x=x_full[start_index:end_index]
            t=t_full[start_index:end_index]
            
            w2=np.add(w,learning_rate*der_err(end_index-start_index,m,t,x,w,lamb))
            a=(reg_sum_square_error(lamb,m,n,t_full,w2,x_full))
            
            while(a>2*err_bound_up):
                learning_rate/=10
                w2=np.add(w,learning_rate*der_err(end_index-start_index,m,t,x,w,lamb))
                a=(reg_sum_square_error(lamb,m,n,t_full,w2,x_full))
            
            w=w2
    return a,learning_rate,w
        
    
    

if method=='gd':
    if lamb==0:
        a,learning_rate,w=gd_batch(data_size,polynomial+1,xdata,tdata,batch_size)
        print('weights=',w)
    else:
        a,learning_rate,w=gd_batch_reg(lamb,data_size,polynomial+1,xdata,tdata,batch_size)
        print('weights=',w)
        
