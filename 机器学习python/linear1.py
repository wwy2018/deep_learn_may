import numpy as np
def gradient(X,y,theta,alpha,iternum,m):
  for i in range(iternum):
    h=np.dot(X,theta)
    theta -= alpha*np.dot(X.T,(h-y)) / m
  return theta,los(X,y,theta,m)
def los(X,y,theta,m):
  h = np.dot(X,theta)
  return np.dot((h-y).T,(h-y)) / (2*m)
def run():
  data=np.genfromtxt('./data/data.txt',delimiter=',',dtype=np.float64)
  m=len(data)
  X=data[:,0:-1]
  X=normalize(X)
  X=np.hstack((np.ones((m,1)),X))
  y=data[:,-1]
  col=data.shape[1]
  theta=np.zeros((col,1))
  y=y.reshape(-1,1)
  iternum=400
  alpha=0.01
  theta, lo=gradient(X,y,theta,alpha,iternum,m)
  print(theta,lo)
def normalize(X):
  xn=np.array(X)
  mu=np.mean(xn,0)
  sigma=np.std(xn,0)
  for i in range(X.shape[1]):
    xn[:,i]=(xn[:,i]-mu[i])/sigma[i]
  return xn
if __name__=='__main__':
  run()