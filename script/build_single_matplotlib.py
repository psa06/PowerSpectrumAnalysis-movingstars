#!/usr/bin/env python

import numpy
import pyfits
import img_scale
import pylab
import os,sys

datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"

fn = os.path.join(datadir, "convolved_map_A-B_fft_thin_disk_20_fml09.fits")
image = os.path.join(datadir, "mapA-B_fml09.png")
sig_fract = 5.0
percent_fract = 0.01

hdulist = pyfits.open(fn)
img_header = hdulist[0].header
img_data_raw = hdulist[0].data
hdulist.close()
width=img_data_raw.shape[0]
height=img_data_raw.shape[1]
print "#INFO : ", fn, width, height
img_data_raw = numpy.array(img_data_raw, dtype=float)
#sky, num_iter = img_scale.sky_median_sig_clip(img_data, sig_fract, percent_fract, max_iter=100)
sky, num_iter = img_scale.sky_mean_sig_clip(img_data_raw, sig_fract, percent_fract, max_iter=10)
print "sky = ", sky, '(', num_iter, ')'
img_data = img_data_raw - sky
min_val = 0.0
print "... min. and max. value : ", numpy.min(img_data), numpy.max(img_data)
'''
new_img = img_scale.sqrt(img_data, scale_min = min_val)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.power(img_data, power_index=3.0, scale_min = min_val)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
'''

new_img = img_scale.log(img_data, scale_min = min_val)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()

sys.exit()

new_img = img_scale.linear(img_data, scale_min = min_val)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.01)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=0.5)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.asinh(img_data, scale_min = min_val, non_linear=2.0)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.histeq(img_data_raw, num_bins=256)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
new_img = img_scale.logistic(img_data_raw, center = 0.03, slope = 0.3)
pylab.imshow(new_img, interpolation='nearest', origin='lower', cmap=pylab.cm.hot)
pylab.axis('off')
pylab.savefig(image)
pylab.clf()
