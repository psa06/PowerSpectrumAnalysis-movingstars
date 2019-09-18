import numpy as np
from astropy.io import fits
import os,sys
from astropy.convolution import convolve_fft
import scipy.signal as ss

#os.system('cp map.fits convoluted_map.fits')

#execfile("useful_functions.py")
#print map.info()


datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"
storagedir = "/run/media/epaic/TOSHIBA EXT/maps/"


'''
G=
lambda_rest=
M_BH =
M =
pi = np.pi
h_p =
c =
Rin =

R = 4.125 light days
R0 = ((45*G*lambda_rest**4*M_BH*M)/(16*pi**6*h_p*c**2))**(1./3.)
'''
def mapdiff(mapA,mapB,r0,comb):

    img = fits.open(mapA)[0]
    map_A = img.data[:, :]


    img = fits.open(mapB)[0]
    map_B = img.data[:, :]

    final_map = map_A/map_B

    hdu = fits.PrimaryHDU(final_map)
    os.chdir(datadir)
    hdu.writeto(storagedir+'Q0158/FML1.0/M0,3/map%s-%s_fml09_R%s_thin_disk.fits'%(comb[0],comb[1],r0))

    return final_map

def half_light_radius(map):
	img = fits.open(map)[0]
	map_d = img.data[:, :]
	tot_lum = np.sum(map_d)
	list_r_half = np.arange(0,200,2)
	map_r = np.zeros(np.shape(map_d))
	xc = int(len(map_d)/2)
	yc = int(len(map_d)/2)

	for lind, line in enumerate(map_d):
		for cind, elt in enumerate(line):
			map_r[cind,lind] = np.sqrt((cind - xc) ** 2 + (lind - yc) ** 2)

	LvsR = []
	for r_half in list_r_half:
		LvsR.append(np.sum(map_d[np.where(map_r<r_half)]))
		print np.where(map_r<r_half)
		print len(np.where(map_r < r_half)[0])
		print len(np.where(map_r < r_half)[1])

	print map_r[246:266,246:266]
	print LvsR
	print list_r_half
	print LvsR[10]
	print LvsR[20]
	print LvsR[40]
	print tot_lum




def convolve( R0, map, model="thin_disk", Rin =0, I0=1):
# map : fits file of the magnification map
# model : string. Can be "thin_disk" (https://arxiv.org/pdf/1707.01908.pdf) or "sersic". It is the model of the light source the mag map will be convoluted with.
# R0 : float. For thin_disk, R0 is R0 (~10^14 cm for 1131 & 0435), for sersic R0 is fwhm. In pixels.
# Rin : float. Useful only for thin_disk
# I0 : float. Useful for both models
#create source profile
	img_name = map.split('map')[2].split('.')[0]
	img = fits.open(map)[0]
	map_d = img.data[:, :]
	macro_mag = np.mean(map_d)
	print macro_mag
	map_d = map_d / macro_mag
	print np.mean(map_d)
	xc = 256
	yc = 256
	xn = 128
	yn = 128
	if model == "thin_disk":
		def getprofilevalue(x, y, xc, yc, I0):
			r = np.sqrt((x-xc)**2 + (y-yc)**2)
			if r<Rin:
				return 0
			else:
				if Rin ==0:
					xi = (r / R0) ** (3. / 4.) * (1) ** (-1. / 4.)
					print "11111"
					print r
					print xi
				else :
					xi = (r/R0)**(3./4.)*(1-np.sqrt(Rin/r))**(-1./4.)
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

	profile = fits.open('source_profile.fits', mode='update')[0]
	gaussian = fits.open('gaussian.fits', mode='update')[0]
	toconv = fits.open('toconv.fits', mode='update')[0]


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
		if x == xc and y ==yc:
			return 1
		else:
			return 0


	for lind, line in enumerate(gdata):
		for cind, elt in enumerate(line):
			gdata[lind][cind] = get2ddiracvalue(cind+1, lind+1, 256, 256)
			if model == "thin_disk":
				pdata[lind][cind] = getprofilevalue(cind + 1, lind + 1, xc, yc, I0)
			if model == "sersic":
				pdata[lind][cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0, reff_pix)
			if model == "wavy":
				pdata[lind][cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0)
			if model == "wavy_hole":
				pdata[lind][cind] = getprofilevalue(cind+1, lind+1, xc, yc, I0)
			if model == "thin_disk&node":
				pdata[lind][cind] = getprofilevalue(cind+1, lind+1, xc, yc,xn,yn, I0)

	print pdata
	print map_d
	#out = scipy.ndimage.filters.convolve(gdata, pdata)
	out = convolve_fft(gdata, pdata)
	#pdata[256][256] = pdata[256][257]
	for lind, line in enumerate(data):
		for cind, elt in enumerate(line):
			data[lind][cind] = out[lind][cind]
	sys.exit()
#	hd=fits.PrimaryHDU(data)
#	hd.writeto("toconv2.fits")

	#toconv.close()
	#profile.close()
	#gaussian.close()



#	if os.path.isfile("convoluted_map.fits"):
#		os.system("rm convoluted_map.fits")

	#toconv = fits.open('toconv.fits', mode='update')[0]
	#kernel = toconv.data[:,:]

	out2 = ss.fftconvolve(map_d, data, mode="valid")
	hdu = fits.PrimaryHDU(out2)

	#os.chdir("")
	hdu.writeto(storagedir+'Q0158/FML1.0/M0,3/convolved_map_%s_fft_%s_%i_fml09.fits'%(img_name,model,R0))

einstein_r = 3.414e16
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = 30000000000*3600*24/cm_per_pxl


dim = 512

x = 0*np.ones(dim)
n = []
for i in range(dim):
	n.append(x)

hdu = fits.PrimaryHDU(n)


#half_light_radius("../toconv.fits")
#sys.exit()

#list_r0 = [2,4,10,15,20,30,40,60,80,100]
list_r0 = [49]
#list_img = ['A','B','A2','B2','A3','B3','A4','A6','A5','A7','A8']
list_img = ['A','B']

#list_comb = [('A3', 'B2'),('A3','B3'),('A4','B2'),('A4','B3'),('A5','B2'),('A5','B3'),('A6','B2'),('A6','B3'),('A7','B3'),('A8','B2'),('A8','B3')]
list_comb = [('A','B')]
#list_comb=[('A2_Re5', 'B2_Re5')]

os.chdir("../")
for img_n in list_img:
	print img_n
	for r0 in list_r0:
		print r0
		convolve(r0, storagedir+"Q0158/FML1.0/M0.3/map%s.fits"%(img_n),"wavy_hole")


for comb in list_comb:
	print comb
	for elem in list_r0:
		print elem
		final_map = mapdiff(storagedir+"Q0158/FML1.0/M0,3/convolved_map_%s_fft_thin_disk_%s_fml09.fits"%(comb[0], elem),storagedir+"Q0158/FML1.0/M0,3/convolved_map_%s_fft_thin_disk_%s_fml09.fits"%(comb[1],elem), elem, comb)


#hdu.writeto("new_canvas.fits")

