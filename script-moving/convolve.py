import numpy as np 
from astropy.io import fits
import os,sys
from astropy.convolution import convolve_fft
import scipy.signal as ss
import glob, shutil

#convolve first 
Res = 512. #map resolution

path = "/home/epfl/pawad/MagnificationMap/"
datadirA = path + "movingstars_magmaps_50days_new_ImgA"  # <----------------- change to new directory 
datadirB = path + "movingstars_magmaps_50days_new_ImgB"  # <----------------- change to new directory
storagedir = path + "/Convolved-Subtracted/"  


#create folder for the convolved maps
def createDir(pathToDir):
	if os.path.exists(pathToDir):
		shutil.rmtree(pathToDir)
		os.mkdir(pathToDir)
	else:
		os.mkdir(pathToDir)


def convolve( R0, map_name, model="thin_disk", Rin =0, I0=1):
# map : fits file of the magnification map
# model : string. Can be "thin_disk" (https://arxiv.org/pdf/1707.01908.pdf) or "sersic". It is the model of the light source the mag map will be convoluted with.
# R0 : float. For thin_disk, R0 is R0 (~10^14 cm for 1131 & 0435), for sersic R0 is fwhm. In pixels.
# Rin : float. Useful only for thin_disk
# I0 : float. Useful for both models
#create source profile
	
	img_name = map_name.split('/')[-1].split('.')[0]
	img_type = img_name.split('_')[1]
	
	img = fits.open(map_name)[0]
	map_d = img.data[:, :]
	macro_mag = np.mean(map_d)
	map_d = map_d / macro_mag
	xc = 256
	yc = 256
	xn = 128
	yn = 128

	if model == "thin_disk":
		def getprofilevalue(x, y, xc, yc, I0):
			r = np.sqrt((x-xc)**2 + (y-yc)**2)
			if x == xc and y ==yc:
				r = 0.00000000001
			if r<Rin:
				return 0
			else:
				if Rin ==0:
					xi = ((r / R0) ** (3. / 4.)) 
					#print r
					#print xi
				else :
					xi = ((r/R0)**(3./4.))*((1-np.sqrt(Rin/r))**(-1./4.))
				return I0/(np.exp(xi)-1)

	if model == "thin_disk&node":
		def getprofilevalue(x, y, xc, yc,xn,yn, I0):
			r = np.sqrt((x-xc)**2 + (y-yc)**2)
			r2 = np.sqrt((x-xn)**2 + (y-yn)**2)
			if r<Rin:
				return 0
			else:
				if Rin ==0:
					xi = (r / R0) ** (3. / 4.) * (1) ** (-1. / 4.)
				else :
					xi = (r/R0)**(3./4.)*(1-np.sqrt(Rin/r))**(-1./4.)
				return I0/(np.exp(xi)-1) + I0/(np.exp((r2/(R0*0.4106))**(3/4))-1)

	if model =="wavy_hole":
		def getprofilevalue(x,y,xc,yc,I0):
			r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
			n = 3
			beta = n*np.pi/(2*R0)
			#if beta*r < np.pi:
			#	return I0*beta/(R0*n*np.pi**2)
			if beta*r < 2*np.pi:
				return I0*beta/(n*np.pi**2)*np.power(np.sin(beta*r),2)/r
			else :
				return 0

	if model =="wavy":
		def getprofilevalue(x,y,xc,yc,I0):
			r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
			n = 3
			beta = n*np.pi/(2*R0)
			if beta*r < np.pi/2:
				return I0*beta/(n*np.pi**2)
			elif beta * r < 2 * np.pi and beta * r > np.pi/2:
				return I0*beta/(n*np.pi**2)*np.power(np.sin(beta*r),2)


	if model == "sersic":
		def getprofilevalue(x, y, xc, yc, I0, reff_pix=0.2, sersic_index=4.0):
			r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
			return I0 * np.exp(-(r / reff_pix) ** (1.0 / sersic_index))



	# canvas is a black 128x128 pixels fits file
	os.system('cp new_canvas.fits toconv.fits')
	os.system('cp new_canvas.fits source_profile.fits')  # non-convolved profile, 128*128
	os.system('cp new_canvas.fits gaussian.fits')  # gaussian, 128*128

	toconv = fits.open('toconv.fits', mode='update')[0]
	profile = fits.open('source_profile.fits', mode='update')[0]
	gaussian = fits.open('gaussian.fits', mode='update')[0]

	data = toconv.data[:,:]
	pdata = profile.data[:,:]
	gdata = gaussian.data[:,:]

	# Read data value
	def getfitsvalue(data, x, y):
		return data[x][y]

	# 2D gaussian profile
	fwhm = R0  # pixels
	sigma = fwhm / 2.355
	def get2dgaussianvalue(x, y, xc, yc):
		return 1.0 / (2 * np.pi * sigma ** 2) * np.exp(-((x - xc) ** 2 + (y - yc) ** 2) / (2 * sigma ** 2))

	def get2ddiracvalue(x,y,xc,yc):
		if x == xc and y == yc:
			return 1
		else:
			return 0


	for lind, line in enumerate(gdata):
		for cind, elt in enumerate(line):
			gdata[lind,cind] = get2ddiracvalue(cind+1, lind+1, 256, 256)
			if model == "thin_disk":
				pdata[lind,cind] = getprofilevalue(cind + 1, lind + 1, xc, yc, I0)
			if model == "sersic":
				pdata[lind,cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0, reff_pix)
			if model == "wavy":
				pdata[lind,cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0)
			if model == "wavy_hole":
				pdata[lind,cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0)
			if model == "thin_disk&node":
				pdata[lind,cind] = getprofilevalue(cind+1, lind+1, xc, yc,xn,yn, I0)

	out = convolve_fft(gdata, pdata)

	#fill the empty canvas with the convolved image
	for lind, line in enumerate(data):
		for cind, elt in enumerate(line):
			data[lind,cind] = out[lind,cind]

	out2 = convolve_fft(map_d, data)

	hdu = fits.PrimaryHDU(out2)
	hdul = fits.HDUList([hdu])
	#save the final result 
	hdul.writeto(storagedir+'convolved_%s/convolved_%s_fft_%s_%i_fml09.fits'%(img_type,img_name,model,R0))



# To create a new empty canvas
##################
'''
# 512 zeros array
dim = int(Res)
x = 0*np.ones(dim)
n = []

#list of lists of zeros
for i in range(dim):
	n.append(x)

hdu = fits.PrimaryHDU(n)
hdul = fits.HDUList([hdu])
hdul.writeto('new_canvas.fits')
'''
##################

#create output directories
name_A = "convolved_A"
name_B = "convolved_B"

createDir(storagedir+name_A)
createDir(storagedir+name_B)

#group all images to be convolved
list_r0 = [15]
list_img1 = glob.glob(datadirA+"/*.fits")
list_img2 = glob.glob(datadirB+"/*.fits")

list_img = list_img1 + list_img2


#convolve
for i, img_n in enumerate(list_img):	
	for r0 in list_r0:
		print i
		convolve(r0,img_n,"thin_disk")













