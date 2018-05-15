import numpy as np
import os
def unitFun(w,b,x):
  eq=w*x+b
  return eq
# using forloop
def los(x,y,l,w,b):
  cos=0
  for i in range(l):
    cos+= (y[i]-(w*x[i]+b))**2 / (2*l)
  return cos
# with matrix
def nplos(x,y,l,w,b):
  return np.sum(np.power(unitFun(w,b,x)-y,2),axis=0) / (2*l)
def gradient(ws,bs,x,y,iternum,rate,l):
  w=ws
  b=bs
  for i in range(iternum):
    [w,b]=step_grad(w,b,x,y,rate,l)
  return [w,b]
def step_grad(w,b,x,y,rate,l):
  wg=0
  bg=0
  for i in range(l):
    wg-=x[i]*(y[i]-unitFun(w,b,x[i])) / l
    bg-=(y[i]-unitFun(w,b,x[i])) / l
  wn=w-(rate*wg)
  bn=b-(rate*bg)
  return [wn,bn]
def run():
  path=os.path.join('data','data.csv')
  data = np.loadtxt(path, delimiter=",")
  l=len(data)
  x=data[:,0]
  y=data[:,1]
  w=0
  b=0
  iternum=1000
  rate=0.0001
  # initial loss before gradient
  print(nplos(x,y,l,w,b))
  [w,b]=gradient(w,b,x,y,iternum,rate,l)\
  # loss,w,b after gradient
  print(nplos(x,y,l,w,b),w,b)
  # test
  print(unitFun(w,b,48))
if __name__=='__main__':
  run()