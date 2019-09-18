from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import matplotlib.pyplot as plt
import os,sys


def fits2png(fitsfile):
    image_file = get_pkg_data_filename(fitsfile)
    image_data = fits.getdata(image_file, ext=0)
    plt.imshow(image_data)
    plt.show()
    plt.savefig(fitsfile+".png")


os.chdir("../data")
fits2png("convoluted_map_A_fft_thin_disk_500.fits")
