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
import matplotlib.tri as tri


execfile('useful_functions.py')


#Useful paths :
datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"
mapdir = "Q0158/FML0.9M0.01/"
storagedir = "/run/media/epaic/TOSHIBA EXT/maps/Q0158/"

#Plot options
font = {'family' : 'normal',
        'size'   : 30}
matplotlib.rc('xtick', labelsize = 30)
matplotlib.rc('ytick', labelsize = 30)
matplotlib.rc('font', **font)


einstein_r_1131= 2.5e16 #Einstein ring of RXJ1131 in cm assuming <M> = 0.3 M_0
einstein_r_03 = 3.414e16 #Einstein ring of QJ0158 in cm assuming <M> = 0.3 M_0
einstein_r_01 = einstein_r_03/np.sqrt(3) #Einstein ring of QJ0158 in cm assuming <M> = 0.1 M_0
einstein_r_001 = einstein_r_03/np.sqrt(30) #Einstein ring of QJ0158 in cm assuming <M> = 0.01 M_0

cm_per_pxl = 20*einstein_r_03/8192 #Find pixel scale assuming the map is 20R_E x 20R_E and 8192 pxl x 8192pxl
ld_per_pxl = cm_per_pxl/(30000000000*3600*24) #Light-day per pixel

#v_source = 31
#day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)

def LSperiodogram(data,time, errdata, frequencies):
    # A periodogram supposed to work on not uniformly sampled data but the tests showed lack of robustness
    power = amlt.lomb_scargle(time, data, errdata, frequencies)
    return power




def good_draw_LC(params, map, time, cm_per_pxl , err_data, add_shut_noise=0):
    #Function to draw a light curve in a microlensing map
    #params is a list composed with the starting coordinaates of the trajectory, its velocity and direction.
    #map is the map used to draw the curve
    #time decides the sampling of the microlensing curve
    #cm_per_pxl gives the scale of the map which is calculated by knowing that a map is 20 R_e x 20 R_e
    # err_data may be used to add experimental shutnoise to the resulting lightcurve if add_shut_noise=1

    # Returns the generated light curve and the coordinates of the starting and ending point of the trajectory, the latter is used only to display the trajectory.

    x_start = params[0]
    y_start = params[1]
    v = params[2]
    angle= params[3]


    #Projecting the velocity on x and y axis
    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)
    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)   

    #Calculating the trajectory of the source in the map
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

    # Checking if the trajectory doesn't go out of the map and gathering the value of the corresponding pixels which give the flux magnification (Hence 2.5*log() to convert in mag).
    if path_x[-1] <= len(map)-1 and path_y[-1] <= len(map)-1 and path_x[-1] >= 0 and path_y[-1] >= 0:
        if add_shut_noise:
            temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),np.random.normal(0, np.mean(err_data), len(path_y)))
        else:
            temp = np.multiply(-2.5, np.log10(map[path_y, path_x]))
        lc = temp - temp[0] * np.ones(len(temp))
        return lc, [path_x[0], path_y[0], path_x[-1], path_y[-1]]



def schuster_periodogram(t, mag, freq):
    # Classic Fourrier transform. Requires uniformly sampled data and not robust enough.
    t, mag, freq = map(np.asarray, (t, mag, freq))
    return abs(np.dot(mag, np.exp(-2j * np.pi * freq * t[:, None])) / np.sqrt(len(t))) ** 2



def power_spectrum(param, map,time,err_data, cm_per_pxl, f, detrend = 'constant', window = 'flattop'):
    # Uses good_draw_LC to draw a lightcurve and then computes its powerspectrum using scipy.signal.periodogram function
    # for f, detrend and window arguments check https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.signal.periodogram.html

    temp = good_draw_LC(param, map, time, err_data, cm_per_pxl)

    if temp is not None:
        lc = temp[0]
        f_cut = 0.01 # highest frequency plotted in days^-1, this is limited by the sampling of the original lightcurve.
        frequency, power = periodogram(lc, f, window=window, detrend=detrend)
        frequency = np.array(frequency[1:])
        power = np.array(power[1:])
        #ax[0].plot(time, lc,alpha=0.7)
        #ax[1].plot(frequency[frequency<f_cut], power[frequency<f_cut], alpha = 0.7)
        return [power, frequency]

def chi2(sim, data, errdata):
    # Measure the quality of fit of the simulted powerspectrum. In this function, a classic chi square formula is implemented
    chi2 = np.sum(np.divide(np.abs(sim - data), errdata))
    return chi2 / len(sim)


#----------------------------------Import Data-----------------------------------------------
f = open(datadir+"/microlensing/data/J0158_Euler_microlensing_upsampled_B-A.rdb","r")
f2 = open(datadir+"/microlensing/data/RXJ1131_ALL_microlensing_upsampled_B-A.rdb","r")

f= f.read()
f=f.split("\n")
data = f[2:]

mjhd = np.array([]) # Time
mag_ml = np.array([]) # Magnitude od microlensing
err_mag_ml = np.array([]) # Error on the magnitude

for i,elem in enumerate(data):
    mjhd = np.append(mjhd,float(elem.split("\t")[0]))
    mag_ml = np.append(mag_ml, float(elem.split("\t")[1]))
    temp = elem.split("\t")[2]
    err_mag_ml= np.append(err_mag_ml,float(temp.split("\r")[0]))

f2= f2.read()
f2=f2.split("\n")
data_2 = f2[2:]

mjhd_2 = np.array([])
mag_ml_2 = np.array([])
err_mag_ml_2 = np.array([])

for i,elem2 in enumerate(data_2):
    mjhd_2 = np.append(mjhd_2,float(elem2.split("\t")[0]))
    mag_ml_2= np.append(mag_ml_2,float(elem2.split("\t")[1]))
    temp = elem2.split("\t")[2]
    err_mag_ml_2= np.append(err_mag_ml_2,float(temp.split("\r")[0]))

# In case your are interested in only a part of the data
#mjhd = mjhd[np.where(mjhd<54250 )]
#mag_ml = mag_ml[np.where(mjhd<54250 )]
#err_mag_ml = err_mag_ml[np.where(mjhd<54250)]

sampling = 1 # Number of days between 2 points

# Fitting a spline through the data curve in order to resample it. Using the pycs package.
new_mjhd = np.arange(mjhd[0], mjhd[-1], sampling)
lc = pycs.gen.lc.factory(mjhd, mag_ml, magerrs=err_mag_ml)
spline = pycs.gen.spl.fit([lc], knotstep=70, bokeps=20, verbose=False)
new_magml = spline.eval(new_mjhd)
new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.var(err_mag_ml), len(new_magml))

# Plot data and spline fit
#fig, ax = plt.subplots(2, 1, figsize=(15, 15))
#ax.errorbar(mjhd, mag_ml, yerr=err_mag_ml,c='r', ls = '')
#ax.plot(mjhd, mag_ml, 'ro', alpha=0.3, label='Microlensing curve')
#ax.plot(new_mjhd, new_magml, label='Spline fit')
#ax.set(xlabel='HJD -2400000.5 [Days]', ylabel=r'$m_A - m_B$')
#ax.legend()
#plt.show()
#fig.savefig('../results/data_fit_spline.png')


# Powerspectrum of the microlensing curve of QJ0158
frequency_spline, power_spline = periodogram(new_magml+  np.random.normal(0, np.mean(err_mag_ml), len(new_mjhd)), 1/sampling,window=window, detrend=detrend)
frequency_spline = np.array(frequency_spline[1:])
power_spline = np.array(power_spline[1:])


# Spline fit and powerspectrum of the microlensing curve of RXJ1131
new_mjhd_2 = np.arange(mjhd_2[0], mjhd_2[-1], sampling)
lc = pycs.gen.lc.factory(mjhd_2, mag_ml_2, magerrs=err_mag_ml_2)
spline_2 = pycs.gen.spl.fit([lc], knotstep=70, bokeps=20, verbose=False)
new_magml_2 = spline_2.eval(new_mjhd_2)

new_err_mag_ml_2 = np.random.normal(np.mean(err_mag_ml_2), np.var(err_mag_ml_2), len(new_magml_2))
frequency_spline_2, power_spline_2 = periodogram(new_magml_2, 1/sampling)
frequency_spline_2 = frequency_spline_2[1:]
power_spline_2=power_spline_2[1:]



if 0:
    # Testing the robustness of the powerspectrum for different sampling of the curve
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    list_sampling = [0.5,1,2,3,4,5,10,50]
    for samp in list_sampling:
        new_mjhd = np.arange(mjhd[0], mjhd[-1], samp)
        new_magml = spline.eval(new_mjhd)
        new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.var(err_mag_ml), len(new_magml))
        frequency_spline, power_spline = periodogram(new_magml, 1/(samp))
        ax[0].plot(frequency_spline[1:], power_spline[1:], '.', label="%s days"% (samp))
        ax[1].plot(new_mjhd, new_magml)
    ax[0].legend()
    ax[0].set(yscale= 'log', xscale = 'log')
    plt.show()
    plt.savefig(resultdir +"robustness_sampling.png")
    sys.exit()


if 0:
    # Creating mock lightcurves using a given magnification map for a given velocity of the source

    list_r0 = [2,4,10,15,20,30,40,60,80,100] #radii of the source in pxl
    n_curves = 100000 #number of generated curves
    n_good_curves = 5000 #minimum number of curves that are not flat
    for r0 in list_r0:
        print r0
        map = storagedir + "FML0.9/M0,3/mapA-B_fml09_R%s.fits" % (r0)
        img = fits.open(map)[0]
        final_map = img.data[:, :]
        v_source =500 #in km.s^-1
        v = v_source * np.ones(n_curves)

        x = np.random.random_integers(200, len(final_map) - 200, n_curves)
        y = np.random.random_integers(200, len(final_map) - 200, n_curves)
        angle = np.random.uniform(0, 2 * np.pi, n_curves)
        params = []
        for i, elem in enumerate(x):
            params.append([x[i], y[i], v[i], angle[i]])

        lc = []
        i = 0
        j=0
        while i<n_good_curves:
            temp= good_draw_LC(params[j], final_map,mjhd, err_mag_ml,(20 * einstein_r_03) / 8192)
            j += 1
            if temp is not None:
                temp = temp[0]
                if np.amax(temp)-np.amin(temp) > 1: # we consider a curve is "not flat" if the difference between it min and max is over 1
                    lc.append(temp)
                    i+=1
                    if i%1000 ==0:
                        print i

        with open(resultdir + 'LCmocks/simLC_A-B_n%s_v%s_R%s_M0,3.pkl' % (
         n_good_curves, v_source, r0), 'wb') as handle:
            pkl.dump((lc), handle, protocol=pkl.HIGHEST_PROTOCOL)

    sys.exit()

if 0:
    #Generating powerspectra of simulated curve
    n_spectrum = 10 #number of curves simulated
    pool = multiprocessing.Pool(2)
    start2 = timeit.default_timer()


    list_comb=[('A','B')] #name of the maps used
    list_v = [100,250,500,1000,2000,5000] #Velocities tested in km.s^-1
    list_r0 = [2,10,20,40] #Radii tested in pxl
    for comb in list_comb:
        print comb
        for r0 in list_r0:
            print r0
            map = storagedir+ "FML0.9/M0,3/map%s-%s_fml09_R%s.fits" % (comb[0],comb[1],r0)
            img = fits.open(map)[0]
            final_map = img.data[:, :]
            for v_source in list_v:
                print v_source
                v = v_source*np.ones(n_spectrum)
                #generating random starting positions and directions
                x = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                y = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                angle = np.random.uniform(0, 2 * np.pi, n_spectrum)

                params = []
                for i, elem in enumerate(x):
                    params.append([x[i], y[i], v[i], angle[i]])


                parrallel_power_spectrum = partial(power_spectrum,map= final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1, cm_per_pxl= (20 * einstein_r_03) / 8192)

                res = pool.map(parrallel_power_spectrum, params)

                #extracting the non None results
                res = filter(None, res)
                res = np.array(res)
                power = res[:,0]
                freq = res[:, 1]

                mean_power = []
                var_power = []
                power=np.array(power)
                for i in range(len(power[0])):
                    mean_power.append(np.mean(power[:,i]))
                    var_power.append(np.var(power[:,i]))

                stop = timeit.default_timer()
                print stop - start2
                #storing the data in pkl file
                with open(resultdir + 'powerspectrum/pkl/spectrum_%s-%s_%s_%s_R%s_M0,01_1.pkl'%(comb[0], comb[1],n_spectrum, v_source, r0), 'wb') as handle:
                    pkl.dump((mean_power,var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)


    sys.exit()



if 1: #Grid of PS
    #Grid plot of powerspectra, each portion contains the powerspectrum for all the velocities for a single radius
    list_comb = [('A3', 'B2')] #name of the maps used to generate the powerspectra
    f_cut = 0.01 #higher frequency considered in days^-1
    r_ref = 15 # Reference radius found in litterature in pxl
    list_r0 = [2,10,20,40, 80,100]
    #list_r0 = [5, 10, 24, 36, 49, 73, 146, 195]
    all_v = [100,300, 500,1000,2000,5000] #Velocities in km.s^-1
    #all_v =[100]
    #list_r0 = [20]
    os.chdir(resultdir+'powerspectrum/pkl/M0.3/A3-B2_flattop') #path to the pkl containing the powerspectra

    #Selecting the frequencies of interest of the powerspectrum
    new_power_spline = power_spline[frequency_spline <= f_cut]
    new_f = frequency_spline[frequency_spline <= f_cut]

    for comb in list_comb:
        ncol = 2 #number of column in the plot
        fig,ax = plt.subplots(int(len(list_r0)/ncol),ncol, figsize=(35,20))
        for i,r in enumerate(list_r0):

            ax[i // ncol][i % ncol].plot(new_f, new_power_spline, "-r", label="Data")
            for v in all_v:
                list_pkl = glob.glob("spectrum_%s-%s_*_%s_R%s_thin-disk_flattop_2.pkl"%(comb[0], comb[1],v,r))
                print list_pkl
                all_power = []
                all_var = []
                for ii,elem in enumerate(list_pkl):
                    print elem
                    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))

                    all_power.append(np.array(mean_power[1:]))
                    all_var.append(np.array(var_power[1:]))
                    freq = np.array(freq[1:])

                mean_power = np.array([])
                var_power = np.array([])
                all_power = np.array(all_power)
                all_var = np.array(all_var)
                for j in range(len(all_power[0])):
                    mean_power=np.append(mean_power, np.mean(all_power[:, j]))
                    var_power=np.append(var_power, max(all_var[:, j]))

                new_f = freq[freq<= f_cut]
                new_mean_power = mean_power[freq<= f_cut]
                new_var_power = var_power[freq<= f_cut]
                ax[i//ncol][i%ncol].plot(new_f, new_mean_power, "--", label=r"$v_e$ = %s km/s"%(v))
                ax[i//ncol][i%ncol].fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)), new_mean_power, alpha = 0.3)
                ax[i//ncol][i%ncol].set(yscale='log',xscale='log', ylim = (0.00001, 10000))#, ylim = (min(power_spline[10:]), max(power_spline)))
                ax[i//ncol][i%ncol].set_title(r"$R_0$ = %s $R_{ref}$"%(round(r/r_ref,1)), fontdict={'fontsize':25})

                if i//ncol==int(len(list_r0)/ncol)-1:
                    locs = ax[i//ncol][i%ncol].get_xticks()
                    locs[1] = frequency_spline[0]

                    temp_lab = ax[i//ncol][i%ncol].get_xticklabels()
                    lab = np.divide(1, locs).astype(int)
                    labels = []
                    for label in lab[1:-1]:
                        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

                    ax[i//ncol][i%ncol].set_xticks(locs[1:-2], minor=False)
                    ax[i//ncol][i%ncol].set_xticklabels(labels, minor = False)

                else:
                    plt.setp(ax[i//ncol][i%ncol].get_xticklabels(), visible=False)

                if i%ncol != 0:
                    plt.setp(ax[i//ncol][i%ncol].get_yticklabels(), visible=False)
        handles, labels = ax[0][1].get_legend_handles_labels()

        ax[0][ncol-1].legend(handles[::-1],labels[::-1],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax[int(len(list_r0)/ncol)-1][1].set(xlabel=r'Frequency (days$^{-1}$)')
        ax[1][0].set(ylabel=r'Power')
        ax[int(len(list_r0)/ncol)-1][ncol-1].xaxis.set_label_coords(-0.2, -0.3)

        fig.subplots_adjust(right=0.83, left = 0.07)
        #plt.show()
        fig.savefig(resultdir + "powerspectrum/png/M0.3/spectrumvsV_%s-%s_thin-disk.png"%(comb[0],comb[1]), dpi = 300)
    sys.exit()

if 0: #PS
    # Plot of the power spectrum for only one radius at a time.
    list_comb = [('A3','B2')]
    f_cut = 0.01
    list_r0 = [2]
    list_v = [5000,1000,500,100]
    os.chdir(resultdir+'powerspectrum/pkl/M0.3/A3-B2_flattop')
    for comb in list_comb:
       for r in list_r0:
           fig = plt.figure(figsize=(10, 10))
           gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.0)
           ax0 = plt.subplot(gs[0])
           ax1 = plt.subplot(gs[1])
           for v_source in list_v:
                list_pkl = glob.glob("spectrum_%s-%s_*_%s_R%s_thin-disk_flattop_2.pkl"%(comb[0], comb[1],v_source,r))
                n_spectrum = int(list_pkl[0].split('_')[2])
                #v_source = []
                #for elem in list_pkl:
                #    v_source.append(int(elem.split('_')[3]))
                #sort = np.argsort(v_source)
                #v_source = np.sort(v_source)
                #list_pkl = [list_pkl[iii] for iii in sort]
                all_pow = []
                all_var = []
                for ii,elem in enumerate(list_pkl):
                    print elem
                    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
                    var_power = np.array(var_power[1:])
                    mean_power= np.array(mean_power[1:])
                    freq = np.array(freq[1:])
                    new_f = freq[freq<= f_cut]
                    new_mean_power = mean_power[freq<= f_cut]
                    new_var_power = var_power[freq<= f_cut]
                    all_pow.append(new_mean_power)
                    all_var.append(new_var_power)
                    ax0.plot(new_f, new_mean_power, "--", label=r"v = %s km/s, "%(v_source))
                    ax0.fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)),
                                    new_mean_power, alpha = 0.3)

                print all_pow
                residuals = all_pow[1]-all_pow[0]
                print residuals.shape
                ax1.plot(new_f, residuals, "--")
                ax1.fill_between(new_f, np.add(residuals, np.sqrt(all_var[0])-np.sqrt(all_var[1])), residuals, alpha=0.3)

           ax0.set(yscale='log',xscale='log')
           ax1.set(yscale='log',xscale='log', ylabel = 'Residuals')
           plt.setp(ax0.get_xticklabels(), visible=False)
           ax0.set_title(r"$R_0$ = %s $R_{ref}$"%(round(r/15,1)), fontdict={'fontsize':18})
           ax0.legend()

           locs = ax1.get_xticks()
           locs[1] = frequency_spline[0]

           temp_lab = ax1.get_xticklabels()
           lab = np.divide(1, locs).astype(int)
           labels = []
           for label in lab[1:-1]:
               labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

                #labels[0]='0'
           ax1.set_xticks(locs[1:-2], minor=False)
           ax1.set_xticklabels(labels, minor = False)

           ax1.legend()
           ax1.set(xlabel=r'Frequency (days$^{-1}$)')
           ax0.set(ylabel=r'Power')
           plt.show()
           fig.savefig(resultdir + "powerspectrum/png/M0.3/spectrumvsV_R%s_A3-B2_flattop.png"%(r),dpi =300)
    sys.exit()




if 0:
    #Computing the powerspectrum of simulated curves
    n_spectrum = 5 #number of curves to simulate
    pool = multiprocessing.Pool(2)
    start2 = timeit.default_timer()

    list_comb=[('A3','B2')]
    list_v = [2000]
    list_r0 = [49]
    for comb in list_comb:
        print comb
        for r0 in list_r0:
            f_cut = 0.01
            print r0
            map = storagedir+ "FML0.9/M0,3/map%s-%s_R%s_wavy-hole_n3.fits" % (comb[0],comb[1],r0)
            img = fits.open(map)[0]
            final_map = img.data[:, :]
            for v_source in list_v:
                print v_source
                v = v_source*np.ones(n_spectrum)

                x = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                y = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                angle = np.random.uniform(0, 2 * np.pi, n_spectrum)

                params = []
                for i, elem in enumerate(x):
                    params.append([x[i], y[i], v[i], angle[i]])


                log_power_spline = np.log10(power_spline[frequency_spline<f_cut])

                lc_spl = pycs.gen.lc.factory(new_mjhd, new_magml, magerrs=new_err_mag_ml)
                spline = pycs.gen.spl.fit([lc_spl], knotstep=1600, bokeps=20, verbose=False)
                magml_spl2 = spline.eval(new_mjhd)
                magml_spl = new_magml - magml_spl2
                frequency_spl, power_spl = periodogram(magml_spl+  np.random.normal(0, np.mean(err_mag_ml), len(new_magml)), 1, window=window,detrend=detrend)
                frequency_spl = np.array(frequency_spl[1:])
                power_spl= np.array(power_spl[1:])

                frequency_spl2, power_spl2 = periodogram(magml_spl2+  np.random.normal(0, np.mean(err_mag_ml), len(new_magml)), 1, window=window, detrend=detrend)
                frequency_spl2 = np.array(frequency_spl2[1:])
                power_spl2 = np.array(power_spl2[1:])

                fig, ax = plt.subplots(2, 1, figsize=(15, 15))
                ax[0].plot(new_mjhd, new_magml, 'r', label="Data")
                #ax[0].plot(new_mjhd, magml_spl2, label="Long time scale variations")
                #ax[0].plot(new_mjhd, magml_spl, label="Short time scale variations")

                for param in params:
                    temp = power_spectrum(param,map= final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1, cm_per_pxl= (20 * einstein_r_03) / 8192, max_deriv=100000*max_deriv_data, n_neighb=n_neighb)


                ax[1].plot(frequency_spline[frequency_spline < f_cut], power_spline[frequency_spline < f_cut], 'r',
                           label="Data")
                #ax[1].plot(frequency_spl2[frequency_spl < f_cut], power_spl2[frequency_spl < f_cut],
                #           label="Long time scale variations")
                #ax[1].plot(frequency_spl[frequency_spl < f_cut], power_spl[frequency_spl < f_cut],
                #           label="Short time scale variations")

                #ax[1].fill_between(freq[freq<f_cut], np.add(mean_power[freq<f_cut], np.sqrt(var_power[freq<f_cut])),
                #                  np.add(mean_power[freq < f_cut], -np.sqrt(var_power[freq < f_cut])), alpha=0.3)
                #ax[2].plot(frequency_spline[frequency_spline < 0.02], np.log10(power_spline[frequency_spline < 0.02]), 'r')
                #ax[2].fill_between(freq[freq < 0.02], np.log10(np.add(mean_power[freq < 0.02], np.sqrt(var_power[freq < 0.02]))),
                #                  np.log10(np.add(mean_power[freq < 0.02], -np.sqrt(var_power[freq < 0.02]))), alpha=0.3)
                #ax[2].plot(np.abs(deriv_data),'k', label='data')
                ax[0].set(xlabel=r'HJD-2400000.5 (days)', ylabel='Magnitude')
                ax[0].xaxis.tick_top()
                ax[0].xaxis.set_label_position('top')
                ax[1].set(xlabel=r'Frequency (days$^{-1}$)', ylabel='Power', yscale='log', xscale='log')
                ax[0].legend()
                #ax[1].legend()
                #ax[2].legend()
                locs = ax[1].get_xticks()
                locs[1] = frequency_spline[0]

                temp_lab = ax[1].get_xticklabels()
                lab = np.divide(1, locs).astype(int)
                labels = []
                for i, elem in enumerate(lab[1:-2]):
                    labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

                ax[1].set_xticks(locs[1:-2], minor=False)
                ax[1].set_xticklabels(labels, minor=False)
                plt.show()
                fig.savefig(resultdir+'powerspectrum/png/necessarynotsufficient.png')
                #ax[2].set(xlabel=r'Frequency (days$^{-1}$)', ylabel='Power')



                mean_power = []
                var_power = []
                power=np.array(power)
                for i in range(len(power[0])):
                    mean_power.append(np.mean(power[:,i]))
                    var_power.append(np.var(power[:,i]))

                stop = timeit.default_timer()
                print stop - start2

                #with open(resultdir + 'powerspectrum/pkl/spectrum_%s-%s_%s_%s_R%s_M0,01_1.pkl'%(comb[0], comb[1],n_spectrum, v_source, r0), 'wb') as handle:
                #    pkl.dump((mean_power,var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)


    sys.exit()


if 0:
    #Calculating mean and standard deviation for the powerspectrum of the data using a high nunmber of noise realisation
    power = []
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    f_cut = 0.2
    for i in range(10000):
        new_magml_2 = new_magml + np.random.normal(0, np.mean(err_mag_ml), len(new_magml))

        frequency_spline, power_spline = periodogram(new_magml_2, 1, window='flattop')
        power.append(power_spline)
        #power.append(lombscargle(new_mjhd, new_magml_2, frequency_spline))

    power = np.array(power)
    mean_p = []
    var_p = []

    for ii in range(len(power[0])):
        mean_p.append(np.mean(power[:,ii]))
        var_p.append(np.var(power[:, ii]))

    frequency_spline = np.array(frequency_spline[1:])
    mean_p = np.array(mean_p[1:])
    var_p = np.array(var_p[1:])
    ax.set_xlim(1 / 4546, 1 / 10)
    ax.plot(frequency_spline[frequency_spline<f_cut], mean_p[frequency_spline<f_cut], "--")
    ax.fill_between(frequency_spline[frequency_spline<f_cut], np.add(mean_p[frequency_spline<f_cut], np.sqrt(var_p[frequency_spline<f_cut])),
                           mean_p[frequency_spline<f_cut], alpha = 0.3)
    ax.set(xlabel=r'Frequency (days$^{-1}$)',ylabel = 'Power', yscale='log', xscale='log')

    #ax.set_title(r"Mean spectrum of the data curve affected by 10000 different photometric noise", fontdict={'fontsize':16})
    #ax.set_title("Same curve with 100000 different realisation of the noise")
    ax.legend(prop={'size':16})
    locs = ax.get_xticks()
    locs[1] = frequency_spline[0]
    locs[-1] = 0.1
    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    labels[-1] = '1'
    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor = False)
    plt.show()

    fig.savefig(resultdir + "powerspectrum/png/spectrum_uncertainty_flattop.png")
    sys.exit()

#### STUDY RES
if 0:
    # Comparing the same trajectories generated by 2 maps with a different resolution. Just a test but not useful anymore
    n_spectrum = 20000
    pool = multiprocessing.Pool(2)
    start2 = timeit.default_timer()
    list_v = [100,250,500]

    map = storagedir+ "FML0.9/M0,3/mapA2_Re5-B2_Re5_fml09_R40.fits"
    img = fits.open(map)[0]
    final_map = img.data[:, :]
    map2 = storagedir + "FML0.9/M0,3/mapA2-B2_fml09_R10.fits"
    img2 = fits.open(map2)[0]
    final_map_2 = img2.data[:, :]

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    inset = plt.axes([0.1, 1, 0.5, 1])
    ip = InsetPosition(ax, [0.2, 0.55, 0.45, 0.45])
    inset.set_axes_locator(ip)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec='0.5')

    for v_source in list_v:
        print v_source
        v = v_source*np.ones(n_spectrum)

        x = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
        y = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
        x_dezoom = x/4+int(len(final_map)*1.5/4)
        y_dezoom = y/4+int(len(final_map)*1.5/4)
        angle = np.random.uniform(0, 2 * np.pi, n_spectrum)


        lc1, arrow1=good_draw_LC([x[0],y[0],v[0],angle[0]], final_map, mjhd, err_mag_ml, (5*einstein_r_03)/8192 )
        lc2, arrow2 = good_draw_LC([x_dezoom[0], y_dezoom[0], v[0], angle[0]], final_map_2, mjhd, err_mag_ml, (20 * einstein_r_03) / 8192)

        #Display of the trajectory in the map
        display_multiple_trajectory([arrow1,arrow2], map, map2)
        plt.plot(mjhd, lc1,label="zoom")
        plt.plot(mjhd, lc2, label = "dezoom")
        plt.legend()
        plt.show()
        sys.exit()


        params = []
        params_2 = []
        for i, elem in enumerate(x):
            params.append([x[i], y[i], v[i], angle[i]])
            params_2.append([x_dezoom[i], y_dezoom[i], v[i], angle[i]])

        parrallel_power_spectrum = partial(power_spectrum,map= final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1, cm_per_pxl = (5*einstein_r_03)/8192 )
        parrallel_power_spectrum_2 = partial(power_spectrum, map=final_map, time=new_mjhd, err_data=new_err_mag_ml, f=1,
                                           cm_per_pxl=(20 * einstein_r_03) / 8192)

        res = pool.map(parrallel_power_spectrum, params)
        res_2 = pool.map(parrallel_power_spectrum_2, params_2)
        res = filter(None, res)
        res = np.array(res)
        power = res[:,0]
        freq = np.array(res[0, 1])
        print freq

        res_2 = filter(None, res_2)
        res_2 = np.array(res_2)
        power_2 = res_2[:, 0]
        freq_2 = res_2[0, 1]
        print freq_2
        mean_power = np.array([])
        var_power = np.array([])
        mean_power_2 = np.array([])
        var_power_2 = np.array([])

        power_2 = np.array(power_2)
        power=np.array(power)
        for i in range(len(power[0])):
            mean_power=np.append(mean_power,np.mean(power[:,i]))
            var_power=np.append(var_power,np.var(power[:,i]))
            mean_power_2=np.append(mean_power_2,np.mean(power_2[:, i]))
            var_power_2=np.append(var_power_2,np.var(power_2[:, i]))



        new_f = freq[freq<= 0.2]
        new_mean_power = mean_power[freq<= 0.2]
        new_var_power = var_power[freq<= 0.2]

        new_f_2 = freq_2[freq_2 <= 0.2]
        new_mean_power_2 = mean_power_2[freq_2 <= 0.2]
        new_var_power_2 = var_power_2[freq_2 <= 0.2]


        ax.plot(new_f, new_mean_power, "--",
                label=r"Zoom ; v = %s km/s" % (v_source))
        ax.fill_between(new_f, np.add(new_mean_power, np.sqrt(new_var_power)),
                        new_mean_power, alpha=0.3)

        ax.plot(new_f_2, new_mean_power_2, "--",
                label=r"Dezoom ; v = %s km/s" % (v_source))
        ax.fill_between(new_f_2, np.add(new_mean_power_2, np.sqrt(new_var_power_2)),
                        new_mean_power_2, alpha=0.3)

        f_cut = freq[freq<= 0.02]
        p_cut = mean_power[freq<= 0.02]
        var_cut = var_power[freq<= 0.02]

        f_cut_2 = freq_2[freq_2 <= 0.02]
        p_cut_2 = mean_power_2[freq_2 <= 0.02]
        var_cut_2 = var_power_2[freq_2 <= 0.02]

        inset.plot(f_cut, p_cut, '--')
        inset.fill_between(f_cut, np.add(p_cut, np.sqrt(var_cut)),
                           p_cut, alpha=0.3)

        inset.plot(f_cut_2, p_cut_2, '--')
        inset.fill_between(f_cut_2, np.add(p_cut_2, np.sqrt(var_cut_2)),
                           p_cut_2, alpha=0.3)


    inset.set(yscale='log')

    ax.set(xlabel=r'Frequency (days$^{-1}$)', ylabel='Power', yscale='log',
           ylim=(min(power_spline[10:]), max(power_spline)))
    ax.legend(prop={'size': 16})
    locs = ax.get_xticks()
    # locs[1] = frequency_spline[0]

    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    labels[0] = '0'
    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor=False)

    locs2 = inset.get_xticks()
    # locs2[1] = frequency_spline[0]
    temp_lab = inset.get_xticklabels()
    lab2 = np.divide(1, locs2).astype(int)
    labels = []
    for i, elem2 in enumerate(lab2[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem2)))

    labels[0] = '0'
    inset.set_xticks(locs2[1:-1], minor=False)
    inset.set_xticklabels(labels, minor=False)
    inset.set_ylim(min(power_spline[10:]), max(power_spline))
    fig.savefig(resultdir + "PowerSvsResolution.png")
    sys.exit()

if 0:
    power = []
    power_2 = []
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    inset = plt.axes([0.1, 1, 0.5, 1])
    ip = InsetPosition(ax, [0.2, 0.55, 0.45, 0.45])
    inset.set_axes_locator(ip)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec='0.5')

    print "youhou"
    for i in range(10000):
        new_magml_3 = new_magml + np.random.normal(0, np.mean(err_mag_ml), len(new_magml))
        new_magml_4 = new_magml_2 + np.random.normal(0, np.mean(err_mag_ml_2), len(new_magml_2))
        freq, power_spline = periodogram(new_magml_3, 1)
        power.append(power_spline[1:])
        freq_2, power_spline_2 = periodogram(new_magml_4, 1)
        power_2.append(power_spline_2[1:])
        #power.append(lombscargle(new_mjhd, new_magml_2, frequency_spline))

    power = np.array(power)
    mean_p = []
    var_p = []
    freq = freq[1:]

    for ii in range(len(power[0])):
        mean_p.append(np.mean(power[:,ii]))
        var_p.append(np.var(power[:, ii]))

    mean_p = np.array(mean_p)
    var_p = np.array(var_p)

    power_2 = np.array(power_2)
    mean_p_2 = []
    var_p_2 = []
    freq_2 = freq_2[1:]

    for ii in range(len(power_2[0])):
        mean_p_2.append(np.mean(power_2[:, ii]))
        var_p_2.append(np.var(power_2[:, ii]))

    mean_p_2 = np.array(mean_p_2)
    var_p_2 = np.array(var_p_2)

    ax.plot(freq[freq<0.2], mean_p[freq<0.2], "--", label = 'Q0158')
    ax.fill_between(freq[freq<0.2], np.add(mean_p[freq<0.2], np.sqrt(var_p[freq<0.2])),
                           mean_p[freq<0.2], alpha = 0.3)
    f_cut = freq[freq<=0.02]
    p_cut = mean_p[freq<=0.02]
    var_cut = var_p[freq<=0.02]

    inset.plot(f_cut, p_cut, '-')
    inset.fill_between(f_cut, np.add(p_cut, np.sqrt(var_cut)),
                           p_cut, alpha=0.3)

    ax.plot(freq_2[freq_2 < 0.2], mean_p_2[freq_2 < 0.2], "--",label='RXJ1131')
    ax.fill_between(freq_2[freq_2 < 0.2], np.add(mean_p_2[freq_2 < 0.2], np.sqrt(var_p_2[freq_2 < 0.2])),
                    mean_p_2[freq_2 < 0.2], alpha=0.3)
    f_cut_2 = freq_2[freq_2 <= 0.02]
    p_cut_2 = mean_p_2[freq_2 <= 0.02]
    var_cut_2 = var_p_2[freq_2 <= 0.02]

    inset.plot(f_cut_2, p_cut_2, '-')
    inset.fill_between(f_cut_2, np.add(p_cut_2, np.sqrt(var_cut_2)),
                       p_cut_2, alpha=0.3)

    inset.set(yscale = 'log')


    ax.set(xlabel=r'Frequency (days$^{-1}$)',ylabel = 'Power', yscale='log')
    ax.set_title(r"Powerspectrum of the data", fontdict={'fontsize':16})
    #ax.set_title("Same curve with 100000 different realisation of the noise")
    ax.legend(prop={'size':16})
    locs = ax.get_xticks()
    locs[1] = freq[0]
    temp_lab = ax.get_xticklabels()
    lab = np.divide(1, locs).astype(int)
    labels = []
    for i, elem in enumerate(lab[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem)))

    ax.set_xticks(locs[1:-1], minor=False)
    ax.set_xticklabels(labels, minor = False)





    locs2 = inset.get_xticks()
    locs2[1] = freq[0]
    temp_lab = inset.get_xticklabels()
    lab2 = np.divide(1, locs2).astype(int)
    labels = []
    for i, elem2 in enumerate(lab2[1:-1]):
        labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(elem2)))

    inset.set_xticks(locs2[1:-1], minor=False)
    inset.set_xticklabels(labels, minor=False)

    plt.show()

    fig.savefig(resultdir + "powerspectrum/png/spectrum_uncertainty_schuster.png")
    sys.exit()

if 1:
    # Plot of the map of chi^2 as function of radius and velocity
    os.chdir(resultdir + 'powerspectrum/pkl/M0.1/A2-B3_flattop')
    list_pkl = glob.glob("*.pkl")
    r_ref =15 #Reference radius in pxl

    # All the variables taht will be extracted from the name of the pkl file
    v_source = []
    reject_v_source=[]
    r0 = np.array([])
    reject_r0 = np.array([])
    chi = []
    reject_chi = []

    better_v = []
    better_r0 = []
    all_chi = np.array([])
    all_r0 = np.array([])
    all_v = []
    maxchi = 10
    f_cut_chi = 0.01
    length = np.array([])
    for i, elem in enumerate(list_pkl):
        print elem
        mean_power, var_power, freq = pkl.load(open(elem, 'rb'))
        #print lenpower
        #length=np.append(length,lenpower)
        var_power = np.array(var_power[1:])
        mean_power = np.array(mean_power[1:])
        temp = chi2(mean_power[frequency_spline<=f_cut_chi],power_spline[frequency_spline<=f_cut_chi], var_power[frequency_spline<=f_cut_chi])
        print temp
        all_chi= np.append(all_chi, temp)
        all_r0= np.append(all_r0,int(elem.split('.')[0].split('_')[4].split('R')[1]))
        all_v.append(int(elem.split('_')[3]))

    z = zip(all_r0, all_v)
    z_unique,inv = np.unique(z, return_inverse=True, axis =0)
    inv_unique = np.unique(inv)
    print z_unique
    print inv_unique

    for i,elem in enumerate(inv_unique):
        temp = np.average(all_chi[np.where(inv == elem)])
        if temp < maxchi:
            chi.append(temp)
            v_source.append(z_unique[i,1])
            r0= np.append(r0,z_unique[i,0])
            if temp <1:
                print "+++++++++++++++"
                print temp
                print z_unique[i, 1]
                print z_unique[i, 0]
                better_v.append(z_unique[i, 1])
                better_r0 = np.append(better_r0, z_unique[i, 0])
        else :
            reject_v_source.append(z_unique[i, 1])
            reject_r0= np.append(reject_r0,z_unique[i, 0])

    print chi
    print v_source
    print r0
    print better_v
    print better_r0

    fig, ax = plt.subplots(1,1, figsize = (13,8))
    sc = ax.scatter(v_source,r0/r_ref, c=chi, cmap = 'Greens_r')
    ax.plot(reject_v_source, reject_r0/r_ref, 'or', label = r"$\chi^2 > %s $"%(maxchi))
    ax.fill_between([110, 750], [1.7, 1.7], [6.7, 6.7], alpha = 0.3, label='Morgan et al.(2012)')
   # ax.scatter(better_v, better_r0/r_ref, s = 90,facecolor='None',edgecolors = "c")

    ax.set(ylabel=r'$R_0$  in units of $R_{ref}$', xlabel=r'$v_e$ [$km\cdot s^{-1}}$]')
    plt.colorbar(sc, label=r"$\chi^2$")
    #ax.text(1000, 1.5, "Best fit : v %s ; R %s" %(v_source[idx], round(r0[idx]/20,1)))

    plt.legend()
    plt.show()
    #fig.savefig(resultdir+"powerspectrum/png/r0vsVvsChi2_FML0,9M0,3_wavy_flattop_A7-B3.png")
    sys.exit()
