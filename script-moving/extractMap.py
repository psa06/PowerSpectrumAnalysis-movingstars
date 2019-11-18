import numpy as np
import matplotlib.pyplot as plt
import os, glob
import shutil

source_scale = 300
SSCALE = 20
lens_Rad = 243.357453971 #corresponds to 300 x 300 (source_scale)
extract_Rad = (SSCALE*lens_Rad)/source_scale














path = os.getcwd()
positions_folder = path + "/positions"

#for each file of positions generated, give x, y columns
def get_pos(position_file):
	data = np.loadtxt(position_file)
	x = data[:,0]
	y = data[:,1]
	m = data[:,2]
	return x,y,m

#draw a circle with a given radius r
def plot_circle(r):
	x = np.arange(-r, r+1, 1./10**4)
	yp = np.sqrt((r**2)-(x**2))
	yn = -np.sqrt((r**2)-(x**2))
	plt.plot(x,yp,'-b')
	plt.plot(x,yn,'-b')

#calculate distance from origin (0,0)
def distance(x,y):
	return np.sqrt((x**2) + (y**2))

'''
#extract the new positions of all stars inside extract_Rad and place them in their own file
name = "positions_extracted"
directory = path+"/positions_extracted/"
if os.path.exists(directory):
	shutil.rmtree(directory)
	os.mkdir(name)
else:
	os.mkdir(name)

positions = sorted(glob.glob(positions_folder+"/*.dat"), key=lambda f: int(filter(str.isdigit,f)))
for j,position_file in enumerate(positions):
	X,Y,mass = get_pos(position_file)
	D = distance(X,Y)
	
	#extract the area
	x = []
	y = []
	m = []
	for i,d in enumerate(D):
		if d <= extract_Rad:
			x.append(X[i])
			y.append(Y[i])
			m.append(mass[i])
	
	#place the extracted positions in new files 	
	f_name = "%s.dat" %str(j)
	f = open(directory+f_name, "w+")
	for n in range(len(x)):
		f.write(str(x[n]) + "    " + str(y[n]) + "    " + str(m[n]) +"\n")
	f.close()

'''


plot_pos("positions/0.dat")
plot_circle(lens_Rad)
plot_circle(extract_Rad)
plt.show()
























