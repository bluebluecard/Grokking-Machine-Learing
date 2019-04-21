import numpy as np
import math
import random
import matplotlib.pyplot as plt

n=100
N=1000

x=np.matrix(np.linspace(-3,3,n)).T
X=np.matrix(np.linspace(-3,3,N)).T

pix=np.pi*x
y=np.sin(pix)/(pix)+0.1*x+0.05*np.random.randn(n,1)

hh=2*np.square(0.3)
t0=np.random.randn(n,1)
e=0.1

for o in range(1,n*1000):
	i=math.ceil(random.random()*n)
	ki=np.exp(-np.square((x-x[i-1]))/hh)
	t=t0-e*ki*(ki.T*t0-y[i-1])
	if np.linalg.norm(t-t0) < 0.000001:
		break
	t0=t

K=np.exp(-(np.tile(np.square(X),(1,n))+np.tile(np.square(x).T,(N,1))-2*X*(x.T))/hh)
F=K*t

plt.plot(X,F,'g-')
plt.plot(x,y,'bo')
plt.show()	
