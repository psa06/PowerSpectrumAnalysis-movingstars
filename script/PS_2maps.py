from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os, sys
import timeit
import multiprocessing
from functools import partial
import pickle as pkl
from astropy.io import fits
import scipy.signal
import glob

# import matplotlib
# from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
#                                                  mark_inset)

host = os.system('hostname')
print host
datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"
storagedir = "/run/media/epaic/TOSHIBA EXT/maps/Q0158/"

# font = {'family' : 'normal',
#        'size'   : 20}

# matplotlib.rc('font', **font)


einstein_r_1131 = 2.5e16  # cm
einstein_r_03 = 3.41e16
einstein_r_01 = einstein_r_03 / np.sqrt(3)
einstein_r_001 = einstein_r_03 / np.sqrt(30)

cm_per_pxl = (20 * einstein_r_03) / 8192
ld_per_pxl = cm_per_pxl / (30000000000 * 3600 * 24)


# day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)

def LSperiodogram(data, time, errdata, frequencies):
    power = amlt.lomb_scargle(time, data, errdata, frequencies)
    return power


def good_draw_LC(params, map, time, err_data):
    x_start = params[0]
    y_start = params[1]
    v = params[2]
    angle = params[3]

    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)

    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)

    if x_start + (time[-1] - time[0]) * v_x <= len(map) and y_start + (time[-1] - time[0]) * v_y <= len(
            map) and x_start + (time[-1] - time[0]) * v_x >= 0 and y_start + (time[-1] - time[0]) * v_y >= 0:
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

        temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),
                      np.random.normal(0, np.mean(err_data), len(path_y)))  # -2.5 log() to convert flux into mag
        lc = temp - temp[0] * np.ones(len(temp))
        return lc


def draw_LC_2maps(params_A, params_B, mapA, mapB, time, err_data):
    lc_A = good_draw_LC(params_A, mapA, time, err_data)
    lc_B = good_draw_LC(params_B, mapB, time, err_data)

    if lc_A is not None and lc_B is not None:
        lc = lc_A - lc_B
        return lc


def schuster_periodogram(t, mag, freq):
    t, mag, freq = map(np.asarray, (t, mag, freq))
    return abs(np.dot(mag, np.exp(-2j * np.pi * freq * t[:, None])) / np.sqrt(len(t))) ** 2


def power_spectrum(param, map, time, err_data, f):
    lc = good_draw_LC(param, map, time, err_data)
    if lc is not None:
        frequency, power = scipy.signal.periodogram(lc, f)
        #        power = scipy.signal.lombscargle(time, lc, f)
        return [power, frequency]


def power_spectrum_2maps(params, map_A, map_B, time, err_data, f):
    param_A = params[0]
    param_B = params[1]
    lc = draw_LC_2maps(param_A, param_B, map_A, map_B, time, err_data)

    if lc is not None:
        frequency, power = scipy.signal.periodogram(lc, f)
        #        power = scipy.signal.lombscargle(time, lc, f)
        return [power, frequency]


def f(x):
    x = x
    return x


# ----------------------------------Import Data-----------------------------------------------

sampling = 1
new_mjhd = np.arange(53601, 58147, sampling)
new_err_mag_ml = np.random.normal(0.008653884816753927, 5.583092113856527e-05, len(new_mjhd))
power_spline = np.array([1.52074311e+03, 9.06592526e+01, 1.75945999e+01, 1.06067646e+01,
                         7.49912860e+00, 4.30770880e+00, 1.45145086e+00, 1.10305442e+00,
                         6.73510371e-01, 3.41822411e-01, 5.26498812e-01, 6.34918046e-01,
                         5.82915467e-01, 2.69384834e-01, 5.75243476e-01, 2.90763001e-01,
                         1.35753666e-01, 2.05492837e-01, 1.86908762e-01, 7.31804094e-02,
                         1.70570766e-01, 1.77605722e-01, 7.43894675e-02, 7.74324174e-02,
                         1.35635794e-01, 1.38526540e-01, 1.15427722e-01, 1.84390285e-01,
                         4.73467461e-02, 9.68420633e-02, 1.09957512e-01, 3.05450474e-02,
                         5.39084993e-02, 5.78878420e-02, 7.50773277e-02, 3.92531588e-02,
                         5.51926275e-02, 4.80699781e-02, 4.69397434e-02, 4.74949568e-02,
                         4.61716670e-02, 3.42199101e-02, 4.66278070e-02, 3.53194945e-02,
                         3.77597483e-02, 3.04772795e-02, 3.83483947e-02, 2.91707602e-02,
                         2.66270446e-02, 3.22889355e-02, 2.47052731e-02, 3.05774833e-02,
                         2.23028260e-02, 2.87057626e-02, 2.12453417e-02, 2.53758963e-02,
                         2.05604249e-02, 2.21325191e-02, 2.07161193e-02, 2.09298707e-02,
                         1.95792894e-02, 1.81327402e-02, 1.97782080e-02, 1.66030710e-02,
                         1.87151846e-02, 1.49350702e-02, 1.84499315e-02, 1.38296619e-02,
                         1.69488323e-02, 1.35664037e-02, 1.55705257e-02, 1.35885219e-02,
                         1.39054515e-02, 1.36423010e-02, 1.24325720e-02, 1.35572380e-02,
                         1.11062780e-02, 1.34092437e-02, 1.03201445e-02, 1.28312071e-02,
                         1.01080587e-02, 1.18656739e-02, 9.94784948e-03, 1.08378908e-02,
                         9.85858472e-03, 9.85849241e-03, 1.01155368e-02, 8.94587451e-03,
                         9.90138835e-03, 8.28462879e-03, 9.70568886e-03, 7.55882617e-03,
                         9.53701913e-03, 7.36735691e-03, 8.81943891e-03, 7.44850276e-03,
                         8.13859040e-03, 7.44534354e-03, 7.38749361e-03, 7.54321756e-03])

frequency_spline = np.array([1.09992164e-04, 2.12908315e-03, 4.14817414e-03, 6.16726513e-03,
                             8.18635612e-03, 1.02054471e-02, 1.22245381e-02, 1.42436291e-02,
                             1.62627201e-02, 1.82818111e-02, 2.03009020e-02, 2.23199930e-02,
                             2.43390840e-02, 2.63581750e-02, 2.83772660e-02, 3.03963570e-02,
                             3.24154480e-02, 3.44345390e-02, 3.64536300e-02, 3.84727209e-02,
                             4.04918119e-02, 4.25109029e-02, 4.45299939e-02, 4.65490849e-02,
                             4.85681759e-02, 5.05872669e-02, 5.26063579e-02, 5.46254488e-02,
                             5.66445398e-02, 5.86636308e-02, 6.06827218e-02, 6.27018128e-02,
                             6.47209038e-02, 6.67399948e-02, 6.87590858e-02, 7.07781768e-02,
                             7.27972677e-02, 7.48163587e-02, 7.68354497e-02, 7.88545407e-02,
                             8.08736317e-02, 8.28927227e-02, 8.49118137e-02, 8.69309047e-02,
                             8.89499956e-02, 9.09690866e-02, 9.29881776e-02, 9.50072686e-02,
                             9.70263596e-02, 9.90454506e-02, 1.01064542e-01, 1.03083633e-01,
                             1.05102724e-01, 1.07121815e-01, 1.09140906e-01, 1.11159997e-01,
                             1.13179088e-01, 1.15198178e-01, 1.17217269e-01, 1.19236360e-01,
                             1.21255451e-01, 1.23274542e-01, 1.25293633e-01, 1.27312724e-01,
                             1.29331815e-01, 1.31350906e-01, 1.33369997e-01, 1.35389088e-01,
                             1.37408179e-01, 1.39427270e-01, 1.41446361e-01, 1.43465452e-01,
                             1.45484543e-01, 1.47503634e-01, 1.49522725e-01, 1.51541816e-01,
                             1.53560907e-01, 1.55579998e-01, 1.57599089e-01, 1.59618180e-01,
                             1.61637271e-01, 1.63656362e-01, 1.65675453e-01, 1.67694544e-01,
                             1.69713635e-01, 1.71732726e-01, 1.73751817e-01, 1.75770908e-01,
                             1.77789999e-01, 1.79809090e-01, 1.81828181e-01, 1.83847272e-01,
                             1.85866363e-01, 1.87885454e-01, 1.89904545e-01, 1.91923636e-01,
                             1.93942727e-01, 1.95961818e-01, 1.97980909e-01, 2.00000000e-01]
                            )

if 1:
    n_spectrum = 10
    start = timeit.default_timer()
    print n_spectrum

    n_cpu = multiprocessing.cpu_count()
    print(n_cpu)
    pool = multiprocessing.Pool(int(n_cpu))

    print "+++++++++++++++"
    x = pool.map(f, range(int(n_cpu)))
    print x
    print "+++++++++++++++"

    list_v = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 5000]
    # list_comb = [('A2', 'B3'), ('A4', 'B4'), ('A3', 'B2'), ('A5', 'B5'), ('A', 'B2'), ('A2', 'B4'), ('A3', 'B5'),('A', 'B'),('A5','B') ]
    #    list_comb = [('A3', 'B2'), ('A3', 'B3'), ('A4', 'B2'), ('A4', 'B3'), ('A5', 'B2'), ('A5', 'B3'), ('A6', 'B2'),('A6', 'B3'),('A7', 'B3'),('A8', 'B2'),('A8', 'B3')]
    list_comb = [('A3', 'B3')]
    list_r0 = [2, 10, 20, 40]
    print list_v
    for comb in list_comb:
        for r0 in list_r0:
            print comb
            print r0
            map_A = storagedir + "FML0.9/M0,3/convolved_map_%s_fft_thin_disk_%s_fml09.fits" % (comb[0], r0)
            img_A = fits.open(map_A)[0]
            final_map_A = img_A.data[:4000, :4000]

            map_B = storagedir + "FML0.9/M0,3/convolved_map_%s_fft_thin_disk_%s_fml09.fits" % (comb[1], r0)
            img_B = fits.open(map_B)[0]
            final_map_B = img_B.data[:4000, :4000]

            for v_source in list_v:
                print v_source
                v = v_source * np.ones(n_spectrum)
                x_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
                y_A = np.random.random_integers(0, len(final_map_A) - 1, n_spectrum)
                angle_A = np.random.uniform(0, 2 * np.pi, n_spectrum)

                x_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
                y_B = np.random.random_integers(0, len(final_map_B) - 1, n_spectrum)
                angle_B = np.random.uniform(0, 2 * np.pi, n_spectrum)

                params = []
                for i, elem in enumerate(x):
                    params.append([[x_A[i], y_A[i], v[i], angle_A[i]], [x_B[i], y_B[i], v[i], angle_B[i]]])
                lc = []
                power = []
                parrallel_power_spectrum = partial(power_spectrum_2maps, map_A=final_map_A, map_B=final_map_B,
                                                   time=new_mjhd, err_data=new_err_mag_ml, f=1 / sampling)

                res = pool.map(parrallel_power_spectrum, params)
                # for param in params:
                #    temp = power_spectrum(param, final_map,new_mjhd,new_err_mag_ml, frequency_spline)
                #    if temp is not None:
                #        power.append(temp[0])
                #        lc.append(temp[1])
                res = filter(None, res)
                res = np.array(res)

                power = res[:, 0]
                freq = res[:, 1]

                mean_power = []
                var_power = []
                power = np.array(power)
                for i in range(len(power[0])):
                    mean_power.append(np.mean(power[:, i]))
                    var_power.append(np.var(power[:, i]))

                stop = timeit.default_timer()
                print(stop - start)
                print len(mean_power)
                print len(var_power)
                print len(freq)
                with open(resultdir + 'powerspectrum/pkl/M0.3/spectrum_%s-%s_%s_%s_R%s_thin-disk_2maps.pkl' % (
                comb[0], comb[1], n_spectrum, v_source, r0), 'wb') as handle:
                    pkl.dump((mean_power, var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)

    sys.exit()

if 0:
    n_spectrum = 100000
    start = timeit.default_timer()
    print n_spectrum

    n_cpu = os.environ['SLURM_JOB_CPUS_PER_NODE']
    print(n_cpu)
    pool = multiprocessing.Pool(int(n_cpu))

    print "+++++++++++++++"
    x = pool.map(f, range(int(n_cpu)))
    print x
    print "+++++++++++++++"

    list_v = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2000, 5000]
    # list_comb = [('A2', 'B3'), ('A4', 'B4'), ('A3', 'B2'), ('A5', 'B5'), ('A', 'B2'), ('A2', 'B4'), ('A3', 'B5'),('A', 'B'),('A5','B') ]
    #    list_comb = [('A3', 'B2'), ('A3', 'B3'), ('A4', 'B2'), ('A4', 'B3'), ('A5', 'B2'), ('A5', 'B3'), ('A6', 'B2'),('A6', 'B3'),('A7', 'B3'),('A8', 'B2'),('A8', 'B3')]
    list_comb = [('A3', 'B2')]
    list_r0 = [24, 36, 49, 73, 100, 146, 195, 10]
    print list_v
    for comb in list_comb:
        for r0 in list_r0:
            print comb
            print r0
            map = datadir + mapdir + "map%s-%s_R%s_wavy-hole_n3.fits" % (comb[0], comb[1], r0)
            img = fits.open(map)[0]
            final_map = img.data[:, :]
            for v_source in list_v:
                print v_source
                v = v_source * np.ones(n_spectrum)
                x = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                y = np.random.random_integers(0, len(final_map) - 1, n_spectrum)
                angle = np.random.uniform(0, 2 * np.pi, n_spectrum)

                params = []
                for i, elem in enumerate(x):
                    params.append([x[i], y[i], v[i], angle[i]])
                lc = []
                power = []
                parrallel_power_spectrum = partial(power_spectrum, map=final_map, time=new_mjhd,
                                                   err_data=new_err_mag_ml, f=1 / sampling)

                res = pool.map(parrallel_power_spectrum, params)
                # for param in params:
                #    temp = power_spectrum(param, final_map,new_mjhd,new_err_mag_ml, frequency_spline)
                #    if temp is not None:
                #        power.append(temp[0])
                #        lc.append(temp[1])
                res = filter(None, res)
                res = np.array(res)

                power = res[:, 0]
                freq = res[:, 1]

                mean_power = []
                var_power = []
                power = np.array(power)
                for i in range(len(power[0])):
                    mean_power.append(np.mean(power[:, i]))
                    var_power.append(np.var(power[:, i]))

                stop = timeit.default_timer()
                print(stop - start)
                print len(mean_power)
                print len(var_power)
                print len(freq)
                with open(resultdir + 'powerspectrum/pkl/M0.3/spectrum_%s-%s_%s_%s_R%s_wavy-hole_n3.pkl' % (
                comb[0], comb[1], n_spectrum, v_source, r0), 'wb') as handle:
                    pkl.dump((mean_power, var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)

    sys.exit()