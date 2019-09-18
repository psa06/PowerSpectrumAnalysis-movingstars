import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os,sys
from scipy.optimize import least_squares
import pickle as pkl
from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import corner

execfile("useful_functions.py")

einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = cm_per_pxl/(30000000000*3600*24)
print ld_per_pxl

v_source = 7000 #km/s
day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)
print day_per_pxl
def draw_LC(x_start, y_start, v_x, v_y, n_points, map):
#    x_start,y_start : Lists of integers. Lists of all the starting coordinates you want to test
#   v_x, v_y : Lists of integers. Lists of all the coordinate of the velocity vector of the source on the lense plane you want to test (for now in pxl/point)
#   n_points : List of integers. List of different length of LC you want to test.
#   map : Array of the convoluted map

    if len(x_start)!= len(y_start) or len(x_start) != len(v_x) or len(x_start)!= len(v_y):
        print "--------------------SHAME ! SHAME ! SHAME !--------------------------"
        print "Your input vectors don't have the same length"
        print "x_start : " + str(len(x_start)) + " y_start : " + str(len(y_start)) + " v_x : " + str(len(v_x)) + " v_y : " + str(len(v_y))
        sys.exit()

#    if x_start + n_points*v_x > len(map_conv)



    time = np.arange(0,n_points)*day_per_pxl


    print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    path_x = []
    path_y = []
    lc = []


    for i,elem in enumerate(x_start):
        if x_start[i]+ n_points*v_x[i] <= len(map) and y_start[i]+ n_points*v_y[i] <=len(map):
            if v_x[i] == 0:
                path_x.append(x_start[i]*np.ones(n_points))
            else:
                path_x.append(np.arange(x_start[i], x_start[i]+ n_points*v_x[i], v_x[i]))
            if v_y[i] ==0:
                path_y.append(y_start[i] * np.ones(n_points))
            else:
                path_y.append(np.arange(y_start[i], y_start[i]+ n_points*v_y[i], v_y[i]))

    for i,elem in enumerate(path_y):
        temp = []
        for j, elem2 in enumerate(path_y[i]):
            temp.append(troncature(-2.5*np.log(map[int(path_x[i][j]), int(path_y[i][j])]),5))
        lc.append(temp) 


        database["trajectory %s"%(i)] = [x_start[i], y_start[i], v_x[i], v_y[i]]

    if len(path_x) == 0:
        sys.exit("No lightcurve to analyse")


    return lc, time




def peak_fitting(elem, time, delta,delta2, model="gaussian", data = False, showplot = False, reverse = False):
    #elem is a single lightcurve e.g. one of the element of the list of lightcurves given by the draw_LC function
    #time are given by the draw_LC function
    #delta : float : Threshold for peak detection, necessary for the use of peakdet_org function
    #model : can be either voigt or gaussian or lorentzian for now
    #showplot : Show individually the fit on each lightcurve simulated ... or not
    #reverse : to fit the minimas

    if data:
        delta = delta2  #if we work with the data, no need to filter the peaks



#    for j,elem in enumerate(lc):      #lc is a list of lists, each sublist is a lightcurve

    if not reverse:
        elem -= np.min(elem)

    if reverse:
        elem = np.abs(elem-np.max(elem))

    max, min, result = peakdet_org(elem, delta)

    real_max, real_min, real_result = peakdet_org(elem, delta2)

    #2 detection of peaks are needed : one with a low delta (around 0.01) in order to make a better fit on the lightcurves. The second peak detection has a bigger delta2 (around 0.06) in order to know which peaks are the interesting ones.

    time_real_max = []
    pos_real_max  =[]
    value_real_max = []
    for i,elem1 in enumerate(real_max):
        time_real_max.append(time[elem1[0]])
        pos_real_max.append(elem1[0])
        value_real_max.append(elem1[1])

#    if len(pos_real_max) >1 and not data:
#        time_real_max = []
#        pos_real_max = []
#        value_real_max = []


    pos_max=[]
    value_max=[]
    for i,elem1 in enumerate(max):
        pos_max.append(time[elem1[0]])
        value_max.append(elem1[1])

    pos_all=[]
    value_all=[]
    for i,elem1 in enumerate(result):
        pos_all.append(time[elem1[0]])
        value_all.append(elem1[1])

    pos_min=[]
    value_min=[]
    for i,elembis in enumerate(min):
        pos_min.append(time[elembis[0]])
        value_min.append(elembis[1])


    for i,elembis in enumerate(pos_max):
        if elembis == 0 or elembis == np.max(time):
            pos_max.remove(elembis)
            value_max.remove(value_max[i])

    for i,elembis in enumerate(pos_min):
        if elembis == 0 or elembis == np.max(time):
            pos_min.remove(elembis)
            value_min.remove(value_min[i])

    for i, elembis in enumerate(pos_real_max):
        if elembis == 0 or elembis == np.max(time):
            pos_real_max.remove(elembis)
            time_real_max.remove(time_real_max[i])
            value_real_max.remove(value_real_max[i])

    #separation of the output of peakdet_org function into 2 vectors : 1 for the values of the peaks and 1 for their position

    if len(pos_real_max) > 0: #We don't use curves that have only very small peaks.
        def fit(time, coeffs):
            #The idea is to fit the data as a sum of function (gaussian, lorentzian ...). The number of terms in the sum of function is determined by the number of peak found by the first detection of peaks.
            if model == "gaussian":
                sum_of_gaussian = 0
                for i in range(len(pos_max)):
                    sum_of_gaussian += coeffs[i * 2 ] * np.ones(len(time)) * np.exp(
                        - ((time - pos_max[i] * np.ones(len(time)))**2 / (2 * np.power(coeffs[i * 2 + 1],2.))))
                return sum_of_gaussian

            if model == "gaussian2":
                sum_of_gaussian = 0

                for i in range(len(pos_max)):
                    sum_of_gaussian +=  coeffs[i * 2] * np.ones(len(time)) * np.exp(
                        - ((time - pos_max[i] * np.ones(len(time))) / (2 * coeffs[i * 2+1])) ** 2)
                return sum_of_gaussian


            if model == "lorentzian":
                sum_of_lorentzian = 0
                for i in range(len(pos_max)):
                    sum_of_lorentzian += coeffs[i*2] / (np.pi *coeffs[2*i+1]*(np.ones(len(time))+ ((time - pos_max[i]*np.ones(len(time)))/coeffs[2*i+1]) ** 2 ))
                return sum_of_lorentzian

            if model == "skewed_gaussian":
                sum_of_skewed = 0
                for i in range(len(pos_max)):
                    sum_of_skewed += skew_gauss(time, coeffs[i*3], pos_max[i], coeffs[i*3+1], coeffs[i*3+2] )
                return sum_of_skewed

            if model == "skewed_gaussian2":
                sum_of_skewed2 = 0
                for i in range(len(pos_max)):
                    sum_of_skewed2 += skewed_gaussian(time, coeffs[i * 3], pos_max[i], coeffs[i * 3 + 1],
                                                coeffs[i * 3 + 2])
                return sum_of_skewed2
        x0 = []

        if model == "gaussian" or model == "gaussian2":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1)    #sigma

        if model == "lorentzian":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1) #gamma

        if model == "skewed_gaussian":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1) #sigma
                x0.append(0.1) #skewness7

        if model == "skewed_gaussian2":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1) #sigma
                x0.append(0.5) #skewness


        print "---------------------------"
        def residuals(coeffs, y, time):
            return fit(time, coeffs) -y


        res = least_squares(residuals, x0,f_scale=0.01, loss = "cauchy", args=(elem, time))
        x = res.x
#            x = leastsq(residuals, x0, args=(elem, time))
#            x = x[0]

        #extraction of all the amplitude and sigma found on the curves
        if model == "gaussian" or model == "gaussian2" or model =="lorentzian":
            amplitude_raw = []
            for elembis in x[0::2]:
                amplitude_raw.append(elembis)
            std_dev_raw = []
            for elembis in x[1::2]:
                std_dev_raw.append(np.abs(elembis))

        if model == "skewed_gaussian" or model =="skewed_gaussian2":
            amplitude_raw = []
            for elembis in x[0::3]:
                amplitude_raw.append(elembis)
            std_dev_raw = []
            for elembis in x[1::3]:
                std_dev_raw.append(np.abs(elembis))

        #Selection of the amplitude and sigmas found only for the peaks detected in the second and more selective peak detection
        amplitude = []
        std_dev = []
        y=[]
        z = []
        width = 2
        if data:
            width = 2
        security = 600
        for i, elembis in enumerate(pos_real_max):

            if (std_dev_raw[i] >0) and (std_dev_raw[i]<security) and ((elembis - width*std_dev_raw[i]/day_per_pxl) >0) and ((elembis + width*std_dev_raw[i]/day_per_pxl) < len(elem)):
                print "+++++++++++++++++++++"
                print std_dev_raw[i]
                #amp = elem[elembis] - (elem[int(elembis - width*std_dev_raw[i]/day_per_pxl)]+elem[int(elembis + width*std_dev_raw[i]/day_per_pxl)])/2
		print value_real_max[i]
                #print amp
                #if amp > delta2/4.0:
                #amplitude.append(amp)
		amplitude.append(value_real_max[i])
                std_dev.append(std_dev_raw[i])
                y.append(elem[int(elembis - width * std_dev_raw[i] / day_per_pxl)])
                y.append(elem[int(elembis + width * std_dev_raw[i] / day_per_pxl)])
                z.append(time[elem.tolist().index(elem[int(elembis - width * std_dev_raw[i] / day_per_pxl)])])
                z.append(time[elem.tolist().index(elem[int(elembis + width * std_dev_raw[i] / day_per_pxl)])])
            else:
                print std_dev_raw[i]
                print "Unable to determine the amplitude"


        #Well ... you get it ;)
	if not data:
		j= array_index(LC,elem.tolist(),reverse)
        if showplot:
            if not data:
                


                arrow= [database["trajectory %s"%(j+1)][0], database["trajectory %s"%(j+1)][1], int(database["trajectory %s"%(j+1)][0] + database["trajectory %s"%(j+1)][2]*n_step), int(database["trajectory %s"%(j+1)][1]+ database["trajectory %s"%(j+1)][3]*n_step)]
                center = [int((arrow[0]+arrow[2])/2), int((arrow[1]+arrow[3])/2)]
                crop_size = 500
                crop = [np.max([center[0]-crop_size, 0]), np.min([center[0]+crop_size, len(map_conv)]), np.max([center[1]-crop_size, 0]), np.min([center[1]+crop_size, len(map_conv)]) ]

                print arrow
                print center
                print crop

                fig, axes = plt.subplots(1, 2, figsize=(14,6))
                display_trajectory(crop, arrow, axes,map=map)
                ax = axes[0]
                if reverse:
                    ax.plot(time, -elem, label = "raw_data")
                    ax.plot(time, -fit(time, res.x), label = "Skewed-gaussian fit")
                    ax.plot(time_real_max, 0*np.ones(len(value_real_max))-value_real_max, "o", label = "Evaluated peak")
                    if len(z) > 0:
                        ax.plot(z, 0*np.ones(len(y))-y , "o", label = "Points use to determine amplitude")

                    #ax.set_ylim((0, -2))
                else:
                    ax.plot(time, elem)
                    ax.plot(time, fit(time, res.x))
                    ax.plot(time_real_max, value_real_max, "o")
                    if len(z)>0:
                        ax.plot(z,y,"o")
                    #ax.set_ylim((0, -2))


                os.chdir("../results")
                plt.show()
                fig.savefig("ex_LC_%s.png" %(j+1) )

            else :
                if reverse:
                    plt.plot(time, -elem)
                    plt.plot(time, -fit(time, x))
                    plt.plot(time_real_max, 0*np.ones(len(value_real_max))-value_real_max, "o")
                    if len(z) > 0:
                        plt.plot(z, 0*np.ones(len(y))-y, "o")
                    #plt.set_ylim((0, -2))


                else:
                    plt.plot(time, elem)
                    plt.plot(time, fit(time, x))
                    plt.plot(time_real_max, value_real_max, "o")
                    if len(z)>0:
                        plt.plot(z,y,"o")
                    #plt.set_ylim((0, -2))


                os.chdir("../results")
                plt.show()
                plt.savefig("data_%s_%s.png" % (reverse, model))


        #Distinction between the lightcurve simulated thanks to draw_LC() and the one coming from the real data. The first one will complete the database.
        if data:
            if reverse:

                return 0*np.ones(len(amplitude))-amplitude, std_dev
            else :
                return amplitude, std_dev

        else:
            if reverse:
	    	print "trajectory %s" % (j)
		print database["trajectory %s" % (j)]
		#db = {}
		#db["trajectory %s"%(j)].append(0*np.ones(len(amplitude))-amplitude)
		#print db
		print amplitude
		f_amp = 0*np.ones(len(amplitude))-amplitude
		print type(f_amp.tolist())
                #database["trajectory %s"%(j)] = f_amp.tolist()
            else :
	    	f_amp = amplitude
                #database["trajectory %s" % (j )].append(amplitude)
            #database["trajectory %s" % (j )].append(pos_real_max)
            #database["trajectory %s" % (j)].append(std_dev)
	    print [f_amp, pos_real_max, std_dev]
            database["trajectory " + str(j)] = database["trajectory " + str(j)] + [ pos_real_max, std_dev]
	    #result["trajectory %s"(str(j))] = [f_amp.tolist(), pos_real_max, std_dev]

    else:
        print "No interesting peaks here"
	


def compare2data(database, amplitude, std_dev, error_margin):
    #lc and time are given by the draw_LC function
    #model : can be either voigt or gaussian for now
    #data : data to which you wanna compare the simulations
    #error_margin : first on the amplitude of the signal and then on the width

    print "oooooooooooooooooooooooooooooooooooooooooooo"
    comparison = {}
    match = {}
    count_winner=0
    count_slow = 0
    count_fast =0
    count_maybe =0
    count_looser = 0
    # First we find which peaks have similar amplitudes in datas and simulations
    for j, elemter in enumerate(amplitude):
        for key, elem in database.iteritems():
            temp = []
            if len(elem) > 4:
                for i, elembis in enumerate(elem[4]):
                    if len(elem[4])>0:
                        if elembis <= elemter+error_margin[0] and elembis >= elemter-error_margin[0]:
                           # print "We maybe have a winner !"
                            comparison[key] = elem
                            temp.append(j)
                            temp.append(i)
                            count_maybe+=1
                        else :
                           # print "Nothing to see here"
                            count_looser+=1

                if len(temp)>0:
                    comparison[key].append(temp)

    # Then we check which one are perfect matchs.
    for key,elem in comparison.iteritems():
        if len(elem) > 4:
            pos_match = elem[7:]
            for elembis in pos_match:
                for i in range(len(elembis)/2):
                    if elem[6][elembis[2*i+1]] <= std_dev[elembis[2*i]] + error_margin[1] and elem[6][elembis[2*i+1]] >= std_dev[elembis[2*i]] - error_margin[1]:
                        print "!!!!!!!!!!!!!!!!!!!!!WE HAVE A WINNER!!!!!!!!!!!!!!!!!!!!!!"
                        count_winner += 1
                        match[key]= database[key]
                        match[key].append("Perfect match for peak %s in sim with peak %s in data" % (
                        elembis[2 * i + 1], elembis[2 * i]))
                    elif elem[6][elembis[2*i+1]] <= std_dev[elembis[2*i]] - error_margin[1]:
                        #print "Slower please"
                        count_slow += 1
     #                   comparison[key].append("Amplitude match for peak %s in sim with peak %s in data" % (elembis[2*i+1], elembis[2*i]))
     #                   comparison[key].append("Too fast : sig_sim vs sig_data : %s vs %s" % (elem[6][elembis[2*i+1]], std_dev[elembis[2*i]]))
                    elif elem[6][elembis[2*i+1]] >= std_dev[elembis[2*i]] + error_margin[1]:
                        #print "Faster please"
                        count_fast += 1
    #                    comparison[key].append("Amplitude match for peak %s in sim with peak %s in data" % (elembis[2*i+1], elembis[2*i]))
    #                    comparison[key].append("Too slow : sig_sim vs sig_data : %s vs %s" % (elem[6][elembis[2*i+1]], std_dev[elembis[2*i]]))




    return comparison, match, count_maybe, count_winner, count_looser, count_slow, count_fast

def histo(db, amplitude, std_dev, filename):

    amp = []
    duration = []
    total = []
    for key,elem in db.iteritems():
	if len(elem) >4:
            for i,elembis in enumerate(elem[4]):
                if elem[6][i] >0 and elem[6][i] < 120 and elem[4][i] >-1 and elem[4][i] <20:
                    temp = []
                    temp.append(elembis)
                    temp.append(elem[6][i])
                    total.append(temp)
                    amp.append(elembis)
                    duration.append(elem[6][i])

    total = np.asarray(total)
 
    toplot=np.column_stack((amp, duration))
    figure = corner.corner(toplot, labels=[r"$Duration$", r"$Amplitude$"], show_titles=True)
      
    fig.savefig(filename)


#######################################  MAIN  #################################################



os.chdir('../data')
map = '../data/convoluted_map_A_fft_thin_disk_30.fits'
#map = 'map_convolved_25.fits'
img = fits.open(map, mode='update')[0]
map_conv = img.data[:, :]

[jds, ml_A, ml_B] = pkl.load(open('ml.pkl', 'rb'))

ml_A = [ml_A]
ml_B = [ml_B]


#[jds, intensity] = pkl.load(open('artificial_peak.pkl', 'rb'))
#intensity = [intensity]

database = {}
result = {}
v_x_int= 1
v_y_int = 0
n_step = 700



sampling = np.arange(0, 8000, 1000)
x_start= np.repeat(sampling, len(sampling))
y_start = np.tile(sampling, len(sampling))

#x_start = np.arange(1000,8000,1000)
#y_start = 1000*np.ones(len(x_start))


v_x = v_x_int*np.ones(len(x_start))
v_y = v_y_int*np.ones(len(y_start))

n_cpu = multiprocessing.cpu_count()
print int(n_cpu)


LC,time = draw_LC(x_start, y_start, v_x, v_y, n_step, map_conv)

print database
delta = 0.06


parallel_peak_fitting = partial(peak_fitting,time = time, delta = delta, delta2 = delta, showplot = False, model="skewed_gaussian2", reverse = True)

#peak_fitting(LC, time, delta, delta2=delta, showplot=True,model = "skewed_gaussian2", reverse = True)
#peak_fitting(LC, time, delta, delta2=delta, showplot=False,model = "skewed_gaussian2", reverse = False)

pool = multiprocessing.Pool(int(n_cpu))
print pool.map(parallel_peak_fitting, LC)

#peak_fitting(LC[13], time, delta, delta2=delta, showplot=False,model = "skewed_gaussian2", reverse = True)

#database = dict(map(lambda kv : (kv[0], parallel_peak_fitting(kv[1])), database.iteritems()))


print "DATABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASE"
print database
print result
exit()

error_margin= [0.005,3]


amplitude, std_dev = peak_fitting(ml_A[0], jds, delta,delta2=delta, data = True, model = "skewed_gaussian2", showplot=False, reverse=True)
histo(database, v_x_int, v_y_int, amplitude, std_dev)


#amplitude2, std_dev2 = peak_fitting(ml_A, jds, delta,delta2=delta, data = True, model = "skewed_gaussian2", showplot=True, reverse=False)
#histo(database, v_x_int, v_y_int, amplitude2, std_dev2)

comp, match, maybe, winner, looser, slower, faster = compare2data(database, amplitude, std_dev, error_margin )
#comp2, match2, maybe2, winner2, looser2, slower2, faster2 = compare2data(database, amplitude2, std_dev2, error_margin )


os.chdir('../results')


file = open("result.txt", 'a')
file.write(str(v_x_int) + "\t" + str(v_y_int)+ "\t" + str(delta)+ "\t" + str(error_margin[0]) + "\t" + str(error_margin[1])+ "\t" + str(looser) + "\t" + str(maybe) + "\t" + str(winner)+ "\n")
file.close()



print "COOOOOOOOOOOOOOOOOOOOOOOOOOOOOOMP"
print comp

print "MAAAAAAAAAAAAAAAAAAAAAAATCH"
print match
print match2
print maybe
print winner
print looser
print slower
print faster
print winner2
print looser2
print slower2
print faster2

print amplitude
print std_dev
print amplitude2
print std_dev2

