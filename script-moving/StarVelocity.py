import numpy as np
import matplotlib.pyplot as plt 
from random import *
from astropy.io import fits
import glob, os
import shutil

path = os.getcwd()
print(path)
thetaE = 3.41E11 #Km

initial_lens_pos = np.loadtxt("lens_pos.dat")
x_init = initial_lens_pos[:,0]
y_init = initial_lens_pos[:,1]
mass = initial_lens_pos[:,2]

hdulist = fits.open("map.fits")
data = hdulist[0].data

pixel_size = (len(data))/(20.*thetaE)
print(pixel_size)
print("data shape", data.shape)

#select time step and time range
dt = 50*24*3600 #sec 50 days
t_tot = 15*365*24*3600 #sec 15 years
ti = np.arange(0,t_tot,dt)
T = len(ti)

#velocity dispersion 
mean, sigma = 0.0, 179.8 # +/- 0.2 km/s
N = len(x_init) #total number of stars

#generate normal distribution 
#randomly select initial velocity for each star
v_mod = []
vx = []
vy = []
vz = []
for i in range(N):
	v = np.random.normal(loc=mean, scale=sigma, size=None)
	theta = np.pi*random()
	phi = 2*np.pi*random()
	
	#project and convert to pixel/s
	vp = v*pixel_size
	v_mod.append(vp)
	vx.append(vp*np.sin(phi)*np.cos(theta))
	vy.append(vp*np.sin(phi)*np.sin(theta))
	vz.append(vp*np.cos(phi))


######################################################################
name = "positions"
directory = path+"/positions/"
if os.path.exists(directory):
	shutil.rmtree(directory)
	os.mkdir(name)
else:
	os.mkdir(name)

x = np.zeros((N,T))
y = np.zeros((N,T))
x[:,0] = x_init
y[:,0] = y_init


#create update formula for new position
for i in range(N):
	for j in range(T-1):
		x[i,j+1] = x[i,j] + vx[i]*dt
		y[i,j+1] = y[i,j] + vy[i]*dt

'''
t = range(T)
plt.plot(t, x[100,:])
plt.show()
'''

#save new positions in file corresponding to the time
for j in range(T):
	col1 = x[:,j]
	col2 = y[:,j] 
	col3 = mass
	t = ti[j]
	f_name = "%s.dat" %t
	f = open(directory+f_name, "w+")
	for n in range(N):
		f.write(str(col1[n]) + "    " + str(col2[n]) + "    " + str(col3[n]) +"\n")
	
	f.close()


#generate magnification map for each time step
'''
fileList = sorted(glob.glob("positions/*.dat"), key=lambda x: int(x.split("/")[1].split(".")[0]))
x100 = np.zeros(T)
for i,f in enumerate(fileList):
	data = np.loadtxt(f)
	x100[i] = data[0,0]


print(x[0,:])
print(" ")
print(x100)

plt.plot(x[0,:], x100)
plt.show()
'''









