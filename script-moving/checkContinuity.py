import numpy as np
import glob
import matplotlib.pyplot as plt

fileList = sorted(glob.glob("positions/*.dat"), key=lambda x: int(x.split("/")[1].split(".")[0]))
dt = 50*24*3600 #sec 50 days
t_tot = 15*365*24*3600 #sec 15 years
t = np.arange(0,t_tot,dt)
 
Ni = 597 #rows 
Nj = len(t) #columns

x = np.zeros((Ni, Nj))
y = np.zeros((Ni, Nj))

for n, f in enumerate(fileList):
	data = np.loadtxt(f)

	xf = data[:,0]
	yf = data[:,1]
	x[:,n] = xf
	y[:,n] = yf

	
N = range(Nj)
plt.plot(N, x[500,:],'.')
plt.show()
