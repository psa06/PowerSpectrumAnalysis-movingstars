import numpy as np 
from astropy.io import fits
import os,sys
from astropy.convolution import convolve_fft
import scipy.signal as ss
import glob, shutil

###### this is only for a given r0, can be changed later ######

path = "/home/epfl/pawad/MagnificationMap/"
storagedir = path + "/Convolved-Subtracted/"

datadirA = storagedir + "/convolved_A/"
datadirB = storagedir + "/convolved_B/"   

#create folder for the final subtracted maps
def createDir(pathToDir):
	if os.path.exists(pathToDir):
		shutil.rmtree(pathToDir)
		os.mkdir(pathToDir)
	else:
		os.mkdir(pathToDir)


#then subtract then maps at each epoch
def mapdiff(mapA,mapB):
	
	S = mapA.split("convolved_map")[1].split("_")
	map_number = S[0]
	final_map_name = "map%s_%s_%s_%s%s_%s"%(S[0],S[2],S[3],S[4],S[5],S[6])	
	
	imgA = fits.open(mapA)[0]
	map_A = imgA.data[:, :]
	imgB = fits.open(mapB)[0]
	map_B = imgB.data[:, :]

	final_map = map_A/map_B

	hdu = fits.PrimaryHDU(final_map)
	hdul = fits.HDUList([hdu])
	hdu.writeto(storagedir+'/subtracted/'+final_map_name)


#create folder to place the maps in
name = "subtracted"
createDir(storagedir + name)

#Arrange the maps of each epoch in pairs
listA = sorted(glob.glob(datadirA +"/*.fits"), key=lambda f: int(filter(str.isdigit,f)))
listB = sorted(glob.glob(datadirB +"/*.fits"), key=lambda f: int(filter(str.isdigit,f)))


#subtract each pair of maps
N = len(listA) #number of pairs --> number of output files 
print("final number of maps: ", N)

for i in range(N):
	print(i)
	mapA = listA[i]
	mapB = listB[i]
	mapdiff(mapA,mapB)













