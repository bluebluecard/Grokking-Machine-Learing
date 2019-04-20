import numpy as np
from numpy.linalg import pinv
import matplotlib.pyplot as plt

n=200
N=10000
x=np.matrix(np.linspace(-3,3,n)).T
X=np.matrix(np.linspace(-3,3,N)).T

pix=np.pi*x
y=np.sin(pix)/(pix)+0.1*x+0.05*np.random.randn(n,1)

p=np.matrix(np.ones([n,31]))
P=np.matrix(np.ones([N,31]))

for j in range(1,16):
	p[:,2*j-1]=np.sin(j/2*x)
	p[:,2*j]=np.cos(j/2*x)
	P[:,2*j-1]=np.sin(j/2*X)
	P[:,2*j]=np.cos(j/2*X)

t = pinv(p).dot(y)
F = P.dot(t)
plt.plot(x,y,'bo')
plt.plot(X,F,'g-')
plt.show()
