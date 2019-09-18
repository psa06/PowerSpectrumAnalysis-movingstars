from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy.signal import detrend, periodogram
from scipy.sparse import csr_matrix
import pycs
from astroML import time_series
from astropy.stats import LombScargle
import sys,os
import pickle as pkl


datadir = "/home/epaic/Documents/Astro/TPIVb/data/"
scriptdir = "/home/epaic/Documents/Astro/TPIVb/script/"
resultdir = "/home/epaic/Documents/Astro/TPIVb/results/"

def ndft(x, f, N):
    """non-equispaced discrete Fourier transform"""
    k = -(N // 2) + np.arange(N)
    return np.dot(f, np.exp(2j * np.pi * k * x[:, np.newaxis]))

def phi(x, n, m, sigma):
    b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
    return np.exp(-(n * x) ** 2 / b) / np.sqrt(np.pi * b)

def phi_hat(k, n, m, sigma):
    b = (2 * sigma * m) / ((2 * sigma - 1) * np.pi)
    return np.exp(-b * (np.pi * k / n) ** 2)

def C_phi(m, sigma):
    return 4 * np.exp(-m * np.pi * (1 - 1. / (2 * sigma - 1)))

def m_from_C_phi(C, sigma):
    return np.ceil(-np.log(0.25 * C) / (np.pi * (1 - 1 / (2 * sigma - 1))))

def nfft(x, f, N, sigma=2, tol=1E-8):
    """Alg 3 from https://www-user.tu-chemnitz.de/~potts/paper/nfft3.pdf"""
    n = N * sigma  # size of oversampled grid
    m = m_from_C_phi(tol / N, sigma)

    # 1. Express f(x) in terms of basis functions phi
    shift_to_range = lambda x: -0.5 + (x + 0.5) % 1
    col_ind = np.floor(n * x[:, np.newaxis]).astype(int) + np.arange(-m, m)
    vals = phi(shift_to_range(x[:, None] - col_ind / n), n, m, sigma)
    col_ind = (col_ind + n // 2) % n
    indptr = np.arange(len(x) + 1) * col_ind.shape[1]
    mat = csr_matrix((vals.ravel(), col_ind.ravel(), indptr), shape=(len(x), n))
    g = mat.T.dot(f)

    # 2. Compute the Fourier transform of g on the oversampled grid
    k = -(N // 2) + np.arange(N)
    g_k_n = fftshift(ifft(ifftshift(g)))
    g_k = n * g_k_n[(n - N) // 2: (n + N) // 2]

    # 3. Divide by the Fourier transform of the convolution kernel
    f_k = g_k / phi_hat(k, n, m, sigma)

    return f_k

def schuster_periodogram(t, mag, freq):
    t, mag, freq = map(np.asarray, (t, mag, freq))
    return abs(np.dot(mag, np.exp(-2j * np.pi * freq * t[:, None])) / np.sqrt(len(t))) ** 2


#f = open(datadir+"/microlensing/data/J0158_Euler_microlensing_upsampled_B-A.rdb","r")
f= f.read()
f=f.split("\n")
data = f[2:]

mjhd = np.array([])
mag_ml = np.array([])
err_mag_ml = np.array([])

for i,elem in enumerate(data):
    mjhd = np.append(mjhd,float(elem.split("\t")[0]))
    mag_ml= np.append(mag_ml,float(elem.split("\t")[1]))
    temp = elem.split("\t")[2]
    err_mag_ml= np.append(err_mag_ml,float(temp.split("\r")[0]))

#mjhd -=mjhd[0]

sampling = 1

new_mjhd = np.arange(mjhd[0],mjhd[-1],sampling)
lc = pycs.gen.lc.factory(mjhd, mag_ml, magerrs=err_mag_ml)
spline=pycs.gen.spl.fit([lc], knotstep =50, bokeps=20,verbose=False)
new_magml = spline.eval(new_mjhd)
new_err_mag_ml = np.random.normal(np.mean(err_mag_ml), np.var(err_mag_ml), len(new_magml))
n_freq = 50
#mag_ml = detrend(mag_ml)
#new_magml=detrend(new_magml)

#pycs.gen.lc.display([lc], [spline])

#x = -0.5 + np.random.rand(1000)
#f = np.sin(10 * 2 * np.pi * x) + np.sin(15 * 2 * np.pi * x) + 20*np.sin(5 * 2 * np.pi * x)+ np.sin(100 * 2 * np.pi * x)

#k = -125 + np.arange(250)
#f_k = ndft(x, f, len(k))

'''
k = np.linspace(-1/(mjhd[1]-mjhd[0]),1/(mjhd[1]-mjhd[0]), n_freq)
new_k = np.linspace(0,1/(mjhd[1]-mjhd[0]), n_freq)
f_k=ndft(mjhd, mag_ml, len(k))
f,Pxx = periodogram(new_magml, sampling)
'''


ls = LombScargle(mjhd, mag_ml, err_mag_ml)
frequency, power = ls.autopower(minimum_frequency = 0, maximum_frequency = 1/50)
print frequency
print power
p_schuster = schuster_periodogram(mjhd, mag_ml, frequency)
#f,Pxx = periodogram(new_magml, sampling)


period_days = 1. / frequency
period_hours = period_days * 24

best_period = period_days[np.argmax(power)]
phase = (mjhd / best_period)




fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax2 = ax[0].twinx()

ax[0].plot(frequency, power, '-k', label="Lomb-Scargle")
ax2.plot(frequency, p_schuster, '-', label = "DFT")
ax[0].legend()
ax2.legend(loc=7)
ax[0].set(xlabel=r'Frequency (days$^{-1}$)')


#ax[1].errorbar(phase, mag_ml, err_mag_ml,
#               fmt='.k', ecolor='gray', capsize=0)
#ax[1].set(xlabel='phase',
#          ylabel='magnitude',
#          title='Phased Data')
#ax[1].invert_yaxis()
#ax[1].text(0.02, 0.03, "Period = {0:.2f} days".format(best_period),
#           transform=ax[1].transAxes)

#inset = fig.add_axes([0.25, 0.6, 0.2, 0.25])
#inset.plot(period_hours, power, '-k', rasterized=True)
#inset.xaxis.set_major_locator(plt.MultipleLocator(1))
#inset.yaxis.set_major_locator(plt.MultipleLocator(0.2))
#inset.set(xlim=(1, 5),
#          xlabel='Period (hours)',
#          ylabel='power')
ax[1].errorbar(mjhd, mag_ml, err_mag_ml,
               fmt='.k', ecolor='gray', capsize=0)
ax[1].set(xlabel='Time (days)',
          ylabel='Magnitude',
          title='Raw Light curve')
ax[1].invert_yaxis()
plt.show()
fig.savefig(resultdir+"Compareperiodogram_J1131_raw_data.png")

sys.exit()
#p = time_series.lomb_scargle(mjhd, mag_ml, err_mag_ml, new_k, generalized=False)

#plt.plot(new_mjhd,new_magml,"+")
#plt.plot(mjhd, mag_ml, "+")
#plt.plot(x,f,"+")
plt.plot(p,k, "+")
plt.show()
plt.plot(k, f_k.real, label='real')
plt.plot(k, f_k.imag,label='imag')
plt.plot(new_k, res)
plt.legend()
plt.show()

plt.plot(f, Pxx)
plt.show()