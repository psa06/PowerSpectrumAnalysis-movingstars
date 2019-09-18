import matplotlib.pyplot as plt
import numpy as np
import os,sys
import pickle as pkl



def gaussian(x, amp, mu, sig):
    return amp*np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


time = np.arange(0, 200, 1)

ampl=0.2
mu1 = 100
sigm=50

plt.plot(time, gaussian(time, ampl, mu1, sigm))

plt.savefig("Example_peak_duration_%s"%(sigm))
plt.show()


intensity = gaussian(time, 0.2, 100, 50)

os.chdir('/home/epaic/Documents/Astro/TPIVb/data')

with open('artificial_peak.pkl', 'wb') as handle:
    pkl.dump([time, intensity], handle, protocol=pkl.HIGHEST_PROTOCOL)


