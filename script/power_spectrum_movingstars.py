from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import timeit
import multiprocessing
from functools import partial
import pickle as pkl
import pycs
from astropy.io import fits
from scipy.signal import lombscargle, periodogram
import astroML.time_series as amlt
import glob
import matplotlib
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
import matplotlib.gridspec as gridspec
from pylab import *
import matplotlib.tri as tridef power_spectrum_multi(lc, f, detrend = 'constant', window = 'flattop'):


def trajectory(params, time, cm_per_pxl ):

    len_map = 7681
    x_start = params[0]
    y_start = params[1]
    v = params[2]
    angle = params[3]
    print params
    # Projecting the velocity on x and y axis
    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)
    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)

    # Calculating the trajectory of the source in the map
    if v_x == 0:
        path_x = x_start * np.ones(len(time))
    else:
        path_x = np.add(np.multiply(np.add(time, -time[0]), v_x), x_start)
    if v_y == 0:
        path_y = y_start * np.ones(len(mjhd))
    else:
        path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

    path_x = path_x.astype(int)
    path_y = path_y.astype(int)
    # Checking if the trajectory doesn't go out of the map
    if path_x[-1] <= len_map - 1 and path_y[-1] <= len_map  - 1 and path_x[-1] >= 0 and path_y[-1] >= 0:
        return [path_x, path_y]
    else:
        return None

def path_chunks(paths, points_per_map):
    path_x = paths[0]
    path_y = paths[1]
    chunks_path_x = [path_x[x:x + points_per_map] for x in xrange(0, len(path_x), points_per_map)]
    chunks_path_y = [path_y[x:x + points_per_map] for x in xrange(0, len(path_y), points_per_map)]

    return [chunks_path_x, chunks_path_y]


def draw_chunkLC(chunks,index, map, err_data, add_shut_noise=0):
    chunk_path_x = np.array(chunks[0][index])
    chunk_path_y = np.array(chunks[1][index])

    if add_shut_noise:
        lc = np.add(np.multiply(-2.5, np.log10(map[chunk_path_y, chunk_path_x])),
                      np.random.normal(0, np.mean(err_data), len(chunk_path_y)))
    else:
        lc = np.multiply(-2.5, np.log10(map[chunk_path_y, chunk_path_x]))
    return lc


def power_spectrum_multi(lc, f, detrend = 'constant', window = 'flattop'):

    frequency, power = periodogram(lc, f, window=window, detrend=detrend)
    frequency = np.array(frequency[1:])
    power = np.array(power[1:])
    return [power, frequency]


#Multimap drawing and powerspectrum
list_maps= ['mapA-B_fml09_R20','mapA-B_fml09_R20','mapA-B_fml09_R20','mapA-B_fml09_R20','mapA-B_fml09_R20','mapA-B_fml09_R20']
points_per_map = int(len(new_mjhd)/len(list_maps))

print points_per_map
n_spectrum = 2000 #number of curves simulated
pool = multiprocessing.Pool(2)
start2 = timeit.default_timer()

list_v = [500] #Velocities tested in km.s^-1
len_map = 7681

for v_source in list_v:
    print v_source
    v = v_source*np.ones(n_spectrum)
    #generating random s    tarting positions and directions
    x = np.random.random_integers(0, len_map - 1, n_spectrum)
    y = np.random.random_integers(0, len_map - 1, n_spectrum)
    angle = np.random.uniform(0, 2 * np.pi, n_spectrum)

    params = []
    for i, elem in enumerate(x):
        params.append([x[i], y[i], v[i], angle[i]])

    parrallel_trajectory = partial(trajectory, time=new_mjhd, cm_per_pxl=cm_per_pxl)
    paths = pool.map(parrallel_trajectory, params)

    paths = filter(None, paths)

    parrallel_chunks=partial(path_chunks, points_per_map=points_per_map)
    path_chunk_res = pool.map(parrallel_chunks, paths)

    lcs = np.empty([len(paths), points_per_map*len(list_maps)])
    for ii,map in enumerate(list_maps):
        map = storagedir + "FML0.9/M0,3/%s.fits" % (map)
        img = fits.open(map)[0]
        final_map = img.data[:, :]


        parrallel_LC = partial(draw_chunkLC,index = ii, map=final_map, err_data = err_mag_ml, add_shut_noise=0)

        temp_lc = pool.map(parrallel_LC, path_chunk_res)
        lcs[:,ii*points_per_map:(ii+1)*points_per_map] = temp_lc

        #plt.plot(new_mjhd[:points_per_map*len(list_maps)], lcs[0],'o')
        #plt.plot(new_mjhd[:points_per_map * len(list_maps)], lcs[1], 'o')
        #plt.plot(new_mjhd[:points_per_map * len(list_maps)], lcs[2], 'o')
        #plt.show()
        #sys.exit()



        parrallel_power_spectrum = partial(power_spectrum_multi,f=sampling, detrend = 'constant', window = 'flattop')
        res = pool.map(parrallel_power_spectrum, lcs)
        res = np.array(res)
        power = res[:,0]
        freq = res[:, 1]
        f_cut = 0.1

        #fig, ax = plt.subplots(2, 1, figsize=(15, 15))
        #ax[0].plot(new_mjhd[:points_per_map*len(list_maps)], lcs[0], alpha=0.7)
        #ax[1].plot(freq[freq < f_cut], power[freq < f_cut], alpha=0.7)
        #plt.show()

        mean_power = np.array([])
        var_power = np.array([])
        power=np.array(power)
        for i in range(len(power[0])):
            mean_power = np.append(mean_power,np.mean(power[:,i]))
            var_power = np.append(var_power,np.var(power[:,i]))


        print type(mean_power)
        freq = freq[0]
        print type(freq<f_cut)

        #fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        #ax.plot(freq[freq < f_cut], mean_power[freq < f_cut], alpha=0.7)
        #ax.fill_between(freq[freq < f_cut], np.add(mean_power[freq < f_cut], np.sqrt(var_power[freq < f_cut])), mean_power[freq < f_cut], alpha=0.3)

        #plt.show()

        stop = timeit.default_timer()
        print stop - start2
        #storing the data in pkl file with the velocity and radius of the source and mean stellar mass of the stars in the mag map.
        with open(resultdir + 'powerspectrum/pkl/spectrum_A-B_%s_%s_R%s_M0,01_1.pkl'%(n_spectrum, v_source, r0), 'wb') as handle:
            pkl.dump((mean_power,var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)


sys.exit()

