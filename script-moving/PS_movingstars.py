import matplotlib.pyplot as plt
import numpy as np
import os,sys
import multiprocessing
from astropy.io import fits
from scipy.signal import lombscargle, periodogram
import glob
import matplotlib


#get all the subtracted magification maps
path = os.getcwd()
datadir = path+"/Convolved-Subtracted/subtracted/"
Maps = sorted(glob.glob(datadir+"/*.fits"), key=lambda f: int(filter(str.isdigit,f)))

#Images details
Res = 512. #pixels
thetaE = 3.41E11 #Km
pixel_size = (Res)/(20.*thetaE) #pixels/Km
err_data = 0 

print(pixel_size)
#define units here
#position imputted in pixels
#velocity inputted in Km/s
def trajectory(params, time, pixel_size):

    x_start = params[0]
    y_start = params[1]
    v_mod = params[2]
    angle = params[3]
    #print params
    
    # Projecting the velocity on x and y axis
    v_x = np.multiply(v_mod, np.cos(angle))
    v_x = np.multiply(v_x, pixel_size)
    v_y = np.multiply(v_mod, np.sin(angle))
    v_y = np.multiply(v_y, pixel_size)

    # Calculating the trajectory of the source in the map
    if v_x == 0:
        path_x = x_start * np.ones(len(time))
    else:
        path_x = np.add(np.multiply(np.add(time, -time[0]), v_x), x_start)
    if v_y == 0:
        path_y = y_start * np.ones(len(time))
    else:
        path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

    path_x = path_x.astype(int)  #uncomment if we want pixel coordinates 
    path_y = path_y.astype(int)  #uncomment if we want pixel coordinates

    # Checking if the trajectory doesn't go out of the map
    if path_x[-1] <= Res-1 and path_y[-1] <= Res-1 and path_x[-1] >= 0 and path_y[-1] >= 0:
        return [path_x, path_y]
    else:
        return None


#divide the trajectory into chunks that correspond to each map
def path_chunks(paths, points_per_map):
    path_x = paths[0]
    path_y = paths[1]
    chunks_path_x = [path_x[x:x + points_per_map] for x in xrange(0, len(path_x), points_per_map)]
    chunks_path_y = [path_y[x:x + points_per_map] for x in xrange(0, len(path_y), points_per_map)]
    chunks = [chunks_path_x, chunks_path_y]
    return chunks

#Caculate the light curve for each map (indexed)
def draw_chunkLC(chunks,index, Map, err_data, add_shot_noise=0):
    #print(index)
    chunk_path_x = np.array(chunks[0][index])
    chunk_path_y = np.array(chunks[1][index])

    if add_shot_noise:
        lc = np.add(np.multiply(-2.5, np.log10(Map[chunk_path_x, chunk_path_y])),
                      np.random.normal(0, np.mean(err_data), len(chunk_path_y)))
    else:
        lc = np.multiply(-2.5, np.log10(Map[chunk_path_x, chunk_path_y]))

    return lc

#Convert the light curves to power spectra  
def power_spectrum_multi(lc, f, detrend = 'constant', window = 'flattop', return_onesided = False):

    frequency, power = periodogram(lc, f, window=window, detrend=detrend)
    frequency = np.array(frequency[1:])
    power = np.array(power[1:])
    return [power, frequency]

###################################################################



n_spectrum = int(1) #number of trajectories drawn

v_mod = 500 #km/s
x_start = np.random.random_integers(0, Res-1, n_spectrum)
y_start = np.random.random_integers(0, Res-1, n_spectrum)
angle = np.random.uniform(0, 2 * np.pi, n_spectrum)

duration = 15*365*24*3600   # 15 years
sample_time = 4*24*3600     # 4 days 
time = np.arange(0,duration, sample_time)

paths = []
for i in range(n_spectrum):
    params = [x_start[i], y_start[i], v_mod, angle[i]]
    path = trajectory(params, time, pixel_size)
    paths.append(path)
print("number of paths: %f" %len(paths))

#save the paths in a file to use in the fixed map code



#for p in path[0]:
#    print(p)

# To plot a trajectory on a chosen map
#hdulist = fits.open("Convolved-Subtracted/subtracted/map108_fft_thin_disk49_fml09.fits")
#Map = hdulist[0].data
#plt.imshow(Map)
#plt.plot(path[0], path[1], color='r')
#plt.show()

####################################################################

points_per_map = int(50/4)   #map is 50 days, sampling every 4 days

#extract the light curve for each path, and store it in LCs
LCs = []
paths = list(filter(None, paths))
for p, path in enumerate(paths):
    print("path #: ", p)  
    chunks = path_chunks(path, points_per_map)
    LC_parts = []
    LC = []
    for i,MagMap in enumerate(Maps):
        Map = fits.open(MagMap)[0].data
        lc = draw_chunkLC(chunks, i, Map, err_data, add_shot_noise=0)
        LC_parts.append(lc)
    
        for elem in lc:
            LC.append(elem)
    LCs.append(LC)

N = len(LCs)
print("number of light curves: %f" %N)

print("Done!")


# For plotting if needed
plt.plot(time[0:len(LCs[N-1])]/(3600*24), LCs[N-1])
#x1 = np.arange(0,50*110,50)
#for x in x1:
#    plt.axvline(x, linestyle='--', color = 'grey')
plt.show()

########################################################################

Power_Spectra = []
for LC in LCs:
   #P = power_spectrum_multi(LC, f=1./4, detrend = 'constant', window = 'flattop')
   #P[0] = np.absolute(P[0])**2
   #Power_Spectra.append(P)
   LC_tild = np.fft.fft(LC)   
   Power_Spectra.append(np.absolute(LC_tild)**2)
print(np.diff(1./time[1:len(Power_Spectra[0])]))
plt.plot(1./time[1:len(Power_Spectra[0])], Power_Spectra[0][1:])

#plt.plot(Power_Spectra[0][1], Power_Spectra[0][0])
plt.show()































