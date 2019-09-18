import numpy as np
import os,sys
import matplotlib.pyplot as plt
from math import factorial
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.visualization import (ZScaleInterval, SqrtStretch, ImageNormalize)
from scipy.special import erf
from matplotlib.patches import Circle
import matplotlib

font = {'family' : 'normal',
        'size'   : 22}
matplotlib.rc('xtick', labelsize = 22)
matplotlib.rc('ytick', labelsize = 22)

matplotlib.rc('font', **font)



from lmfit.models import SkewedGaussianModel

datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"


def peakdet_org(v, delta, x=None):
# code found on https://stackoverflow.com/questions/37931074/peak-detection-and-isolation-in-a-noisy-spectra-1d

#Converted from MATLAB script at http://billauer.co.il/peakdet.html

#Returns two arrays

#function [maxtab, mintab]=peakdet(v, delta, x)
#PEAKDET Detect peaks in a vector
#        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
#        maxima and minima ("peaks") in the vector V.
#        MAXTAB and MINTAB consists of two columns. Column 1
#        contains indices in V, and column 2 the found values.
#
#        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
#        in MAXTAB and MINTAB are replaced with the corresponding
#        X-values.
#
#        A point is considered a maximum peak if it has the maximal
#        value, and was preceded (to the left) by a value lower by
#        DELTA.

# Eli Billauer, 3.4.05 (Explicitly not copyrighted).
# This function is released to the public domain; Any use is allowed.


    maxtab = []
    mintab = []
    resulttab = []

    if x is None:
        x = np.arange(len(v))


    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.inf, -np.inf
    mnpos, mxpos = np.nan, np.nan

    lookformax = True

    for i in range(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append([mxpos, mx])
                resulttab.append([mxpos, mx])
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append([mnpos, mn])
                resulttab.append([mxpos, mx])
                mx = this
                mxpos = x[i]
                lookformax = True

    return maxtab, mintab, resulttab


def display_trajectory(crop_coord , arrow_coord ,axes, map, scale = "logZscale"):
    image_file = get_pkg_data_filename(map)
    image_data = fits.getdata(image_file, ext=0)
#    image_data = np.swapaxes(image_data,0,1)
    image_data = image_data[np.min([crop_coord[0],crop_coord[1]]):np.max([crop_coord[0],crop_coord[1]]),np.min([crop_coord[2],crop_coord[3]]):np.max([crop_coord[2],crop_coord[3]])]
    real_arrow = [arrow_coord[0] - crop_coord[0], arrow_coord[1] - crop_coord[2],
                  arrow_coord[2] - crop_coord[0] - (arrow_coord[0] - crop_coord[0]),
                  -crop_coord[2] + arrow_coord[3] - (arrow_coord[1] - crop_coord[2])]
    #print real_arrow
    ax = axes[1]
    if scale =="linZscale":
        norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.1),
                          stretch=SqrtStretch())

        im=ax.imshow(image_data, cmap='gray', norm = norm)
        plt.colorbar(im, ax=ax)

    if scale =="logZscale":
        new_image_data = -2.5*np.log10(abs(image_data))
        norm = ImageNormalize(new_image_data-np.min(new_image_data), interval=ZScaleInterval(contrast=0.25), stretch=SqrtStretch())
        im=ax.imshow(new_image_data - np.min(new_image_data), aspect= 'auto', cmap='gray')
        plt.colorbar(im,ax=ax)

    ax.arrow(real_arrow[1], real_arrow[0], real_arrow[3], real_arrow[2], head_width=50,
                 head_length=50, fc='g', ec='g')


def display_trajectory_bis(arrow_coord, map):
    arrow = [arrow_coord[0], arrow_coord[1], int(arrow_coord[2]-arrow_coord[0]), arrow_coord[3]-arrow_coord[1]]
    print arrow
    image_file = get_pkg_data_filename(map)
    image_data = fits.getdata(image_file, ext=0)
    #    image_data = np.swapaxes(image_data,0,1)
    real_arrow = [arrow_coord[0], arrow_coord[1], arrow_coord[2], arrow_coord[3]]
    new_image_data = -2.5 * np.log(abs(image_data))
    norm = ImageNormalize(new_image_data - np.min(new_image_data), interval=ZScaleInterval(contrast=0.25),
                          stretch=SqrtStretch())
    im = plt.imshow(new_image_data - np.min(new_image_data), aspect='auto', cmap='gray')
    plt.colorbar(im)

    plt.arrow(arrow[0], arrow[1], arrow[2], arrow[3] , head_width=50,
              head_length=50, fc='g', ec='g')
    plt.show()








def display_multiple_trajectory(arrow_coord, map,map_2, scale = "logZscale"):
    image_file = get_pkg_data_filename(map)
    image_data = fits.getdata(image_file, ext=0)
    image_file_2 = get_pkg_data_filename(map_2)
    image_data_2 = fits.getdata(image_file_2, ext=0)
    fig,ax = plt.subplots(1,2, figsize=(15,15))
    #print real_arrow
    if scale =="linZscale":
        norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.1),
                          stretch=SqrtStretch())

        im=ax[0].imshow(image_data, cmap='gray', norm = norm)
        im_2 = ax[1].imshow(image_data_2, cmap='gray', norm=norm)
        plt.colorbar(im, ax=ax)

    if scale =="logZscale":
        new_image_data = 2.5*np.log10(abs(image_data))
        new_image_data_2 = 2.5 * np.log10(abs(image_data_2))
        norm = ImageNormalize(new_image_data-np.min(new_image_data), interval=ZScaleInterval(contrast=0.25), stretch=SqrtStretch())
        im=ax[0].imshow(new_image_data - np.min(new_image_data), aspect= 'auto', cmap='gray')
        im_2 = ax[1].imshow(new_image_data_2 - np.min(new_image_data_2), aspect='auto', cmap='gray')
        plt.colorbar(im)
    #for i,elem in enumerate(arrow_coord):
    #    real_arrow = [elem[0], elem[1],
    #                  elem[2]- elem[0],
    #                  elem[3] - (elem[1])]
    #    plt.arrow(real_arrow[0], real_arrow[1], real_arrow[2], real_arrow[3], head_width=50,
     #                head_length=50, fc='g', ec='g')
    real_arrow = [arrow_coord[0][0], arrow_coord[0][1], arrow_coord[0][2]- arrow_coord[0][0],arrow_coord[0][3] - (arrow_coord[0][1])]
    real_arrow_2 = [arrow_coord[1][0], arrow_coord[1][1], arrow_coord[1][2] - arrow_coord[1][0],
                  arrow_coord[1][3] - (arrow_coord[1][1])]
#    ax.arrow(real_arrow[0], len(image_data)- real_arrow[1], real_arrow[2],-real_arrow[3], head_width=50, head_length=50, fc='g', ec='g')
#    print real_arrow
    ax[0].arrow(real_arrow[0], real_arrow[1], real_arrow[2], real_arrow[3], head_width=50,head_length=50, fc='g', ec='g')
    ax[1].arrow(real_arrow_2[0], real_arrow_2[1], real_arrow_2[2], real_arrow_2[3], head_width=50, head_length=50, fc='g',
                ec='g')
    plt.show()

def convert_fits2png(fitsfile,png, scale = "logZscale"):
    image_file = get_pkg_data_filename(fitsfile)
    image_data = fits.getdata(image_file, ext=0)
    #print real_arrow
    if scale =="linZscale":
        norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.1),
                          stretch=SqrtStretch())

        im=plt.imshow(image_data, cmap='gray', norm = norm)

    if scale =="logZscale":
        new_image_data = -2.5*np.log(abs(image_data))
        norm = ImageNormalize(new_image_data-np.min(new_image_data), interval=ZScaleInterval(contrast=0.25), stretch=SqrtStretch())
        im=plt.imshow(new_image_data - np.min(new_image_data), aspect= 'auto', cmap='gray')
    plt.show()
    im.savefig(png)




def skewed_gaussian(x, A,mu, sigma, gamma):
    return A/(sigma*np.sqrt(2*np.pi))*np.exp(-((x-mu)**2)/(2*sigma**2))*(1+erf(gamma*(x-mu)/(sigma*np.sqrt(2))))


def array_index(LC, lc):
    for i, elem1 in enumerate(LC):
	#elem1 = np.abs(elem1-np.max(elem1))
        if elem1 ==lc:
            return i
       # if elem1.tolist() == lc.tolist():


def troncature(x, n):
    x = np.multiply(x,10**n)
    x = x.astype(int)
    return np.divide(x,10**n).astype(float)


#def fit_histo():
#    xvals, yvals =
#
#    model = SkewedGaussianModel()
#
#    # set initial parameter values
#    params = model.make_params(amplitude=10, center=0, sigma=1, gamma=0)
#
#    # adjust parameters  to best fit data.
#    result = model.fit(yvals, params, x=xvals)
#
#    print(result.fit_report())
#    pylab.plot(xvals, yvals)
#    pylab.plot(xvals, result.best_fit)


    #print array_index(x,y)
#crop = [100, 1000, 100, 1000]
#arrow = [200, 200,300,300]
#display_trajectory(crop, arrow)


def display_map( map, scale = "logZscale"):
    img = fits.open(map)[0]
    image_data = img.data[:,:]
    image_data = img.data[:, :]/np.mean(image_data)
    for i,line in enumerate(image_data):
        for j,col in enumerate(line):
            image_data[i][j]=max([col, 0.00001])

    print len(image_data)
    #image_data= image_data/((1-0.72)**2-1.03**2)
    fig, ax = plt.subplots(1, figsize=(10,10))
    ax.set_aspect('equal')

    if scale =="linZscale":
        norm = ImageNormalize(image_data, interval=ZScaleInterval(contrast=0.01),
                          stretch=SqrtStretch())

        im=ax.imshow(image_data, cmap='gray', norm = norm)
        plt.colorbar(im, ax=ax)

    if scale =="logZscale":
        new_image_data = 2.5*np.log(image_data)
        norm = ImageNormalize(new_image_data, interval=ZScaleInterval(contrast=0.000001),
                              stretch=SqrtStretch())
        im=ax.imshow(new_image_data - np.min(new_image_data), aspect= 'auto', cmap='gray')
        #plt.colorbar(im,ax=ax)

    circ = Circle((1000, 3000), 20, ec = "red", fc = None, fill = False)
    circ2 = Circle((1000,3000), 250, ec = "green",fc = None, fill = False)
    #ax.add_patch(circ)
    #ax.add_patch(circ2)
    #plt.axis('off')
    locs2 = ax.get_xticks()
    print locs2
    # locs2[1] = frequency_spline[0]
    temp_lab = ax.get_xticklabels()

    lab2 = np.round(np.multiply(0.00244140625,locs2[1:-1]),1)
    print lab2
    labels = []
    ax.set(xlabel = r'$\theta_E$', ylabel = r'$\theta_E$')
    ax.set_xticks(locs2[1:-1], minor=False)
    ax.set_yticks(locs2[1:-1], minor=False)
    ax.set_xticklabels(lab2, minor=False)
    ax.set_yticklabels(lab2, minor=False)
    plt.show()
    fig.savefig('mapA-B.png')

display_map('/run/media/epaic/TOSHIBA EXT/maps/Q0158/FML0.9/M0,3/mapA-B_fml09_R20.fits')
#display_map('../toconv.fits')

def weighted_quantile(values, quantiles, sample_weight=None, values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of initial array
    :param old_style: if True, will correct output to be consistent with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.add(np.cumsum(sample_weight),np.multiply(-0.5,sample_weight))
    weighted_quantiles = weighted_quantiles.astype(float)
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)

    return np.interp(quantiles, weighted_quantiles, values)