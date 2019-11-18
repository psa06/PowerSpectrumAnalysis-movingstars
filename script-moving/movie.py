import imageio
import glob
import matplotlib.pyplot as plt
import f2n as f2n
from astropy.io import fits
import os

path = os.getcwd()

def fits2png(fitsfile, name):
	image = path + "/"+name+".png"
	data = f2n.fromfits(path + "/" + str(fitsfile), hdu=0, verbose=True)
	data.setzscale(z1='ex',z2='ex')
	data.makepilimage(scale="log", negative=False)
	data.tonet(image)


#change all fits files to png images
fitsnames = sorted(glob.glob('*.fits'), key=lambda f: int(filter(str.isdigit,f)))
for i,file in enumerate(fitsnames):
   fits2png(file, str(i))
 
#create movie with png images
pngnames = sorted(glob.glob('*.png'), key=lambda x: int(os.path.splitext(x)[0]))
images = []
for image in pngnames:
	images.append(imageio.imread(image))

imageio.mimsave('movie.gif', images, fps=5)

