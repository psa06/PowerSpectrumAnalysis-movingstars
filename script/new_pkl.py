import pickle as pkl
import numpy as np
import os,sys
import glob

os.chdir('/scratch/paic/results/powerspectrum/pkl/M0.3/wavy/')
list_pkl = glob.glob("*.pkl")

for elem in list_pkl:
    mean_power, var_power, freq = pkl.load(open(elem, 'rb'))

    os.system("rm %s"%(elem))
    with open('/scratch/paic/results/powerspectrum/pkl/M0.3/wavy/%s' % (elem), 'wb') as handle:
        pkl.dump((mean_power, var_power, freq[0]), handle, protocol=pkl.HIGHEST_PROTOCOL)
