import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os,sys
from scipy.optimize import least_squares
import pickle as pkl
import multiprocessing
from functools import partial
import corner #for the plots

execfile("useful_functions.py")


# The idea of the program is to create a dictionnary (database) initially containing the starting coordinates and velocities of every curve ; the value, position and width of the peak are then added to it.


workdir = "../"

#Some calculations to convert physical distances in pixel distances
einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = cm_per_pxl/(30000000000*3600*24)

v_source = 7000 #km/s
day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)



def draw_LC(x_start, y_start, v_x, v_y, n_points, map):

#    x_start,y_start : Lists of integers. Lists of all the starting coordinates you want to test
#   v_x, v_y : Lists of integers. Lists of all the coordinate of the velocity vector of the source on the lense plane you want to test (for now in pxl/point)
#   n_points : Integer. Length of LC you want to simulate.
#   map : Array of the convoluted map

# One element of x_start, y_start, v_x and v_y is used to create a single lightcurve. If those parameters are lists instead of integers you will get list of lightcurves stored in lc. time is a single list valid for every lightcurve.


    if len(x_start)!= len(y_start) or len(x_start) != len(v_x) or len(x_start)!= len(v_y):
        print "--------------------SHAME ! SHAME ! SHAME !--------------------------"
        print "Your input vectors don't have the same length"
        print "x_start : " + str(len(x_start)) + " y_start : " + str(len(y_start)) + " v_x : " + str(len(v_x)) + " v_y : " + str(len(v_y))
        sys.exit()



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
            temp.append(troncature(-2.5*np.log(map[int(path_x[i][j]), int(path_y[i][j])]),5))  #-2.5 log() to convert flux into mag
        lc.append(temp) 


        database["trajectory %s"%(i)] = [x_start[i], y_start[i], v_x[i], v_y[i]]

    if len(path_x) == 0:
        sys.exit("No lightcurve to analyse")


    return lc, time




def peak_fitting(elem, time, delta, model="gaussian", data = False, showplot = False, reverse = False):
    #elem is a single lightcurve e.g. one of the element of the list of lightcurves given by the draw_LC function
    #time is given by the draw_LC function
    #delta : float : Threshold for peak detection, necessary for the use of peakdet_org function
    #model : can only be skewed gaussian or gaussian for now
    #showplot : Show individually the fit on each lightcurve simulated ... or not
    #reverse : to fit the minimas
    
    # This function is meant to be paralellized, see parallel_peak_fitting later.
    # The idea of the function is to take one light curve, find the local maximas using peakdet_org (see useful_functions), fit the maximas with a sum of whatever function you want and add the results to the already existing dictionnary. 
    #It is possible to return the result for a single curve as separated lists for amplitude and width (if you treat real data instead of simulation for example) by switching the flag "data" to True. 
    #The number of terms in the sum is determined by the number of maximas detected. 
    

    if not reverse:
        elem = np.abs(elem-np.max(elem))
	elem = elem.tolist()
    else:
        elem -= np.max(elem)
	elem = elem.tolist()
	
    max, min, result = peakdet_org(elem, delta)
    

 #separation of the output of peakdet_org function into 2 vectors : 1 for the values of the peaks and 1 for their position
    pos_max=[]
    value_max=[]
    time_max= []
    #print "oooooooooooooooooooooooooo"
    #print max
    for i,elem1 in enumerate(max):
    	time_max.append(time[elem1[0]])
        pos_max.append(elem1[0])
        value_max.append(elem1[1])
	
    #print time_max
    #print value_max

    for i,elembis in enumerate(pos_max):
        if elembis == 0 or elembis == np.max(time):
            pos_max.remove(elembis)
            value_max.remove(value_max[i])
	    time_max.remove(time_max[i])
   

    if len(pos_max) > 0: #We don't fit curves that have only very small peaks.
        def fit(time, coeffs):
            if model == "gaussian":
                sum_of_gaussian = 0
                for i in range(len(pos_max)):
                    sum_of_gaussian += coeffs[i * 2 ] * np.ones(len(time)) * np.exp(
                        - ((time - pos_max[i] * np.ones(len(time)))**2 / (2 * np.power(coeffs[i * 2 + 1],2.))))
                return sum_of_gaussian

            if model == "skewed_gaussian":
                sum_of_skewed = 0
                for i in range(len(pos_max)):
                    sum_of_skewed += skewed_gaussian(time, coeffs[i*3], pos_max[i], coeffs[i*3+1], coeffs[i*3+2] )
                return sum_of_skewed


# x0 is the list containing the starting parameter for the fit 
        x0 = []

        if model == "gaussian":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1)    #sigma


        if model == "skewed_gaussian":
            for i, elembis in enumerate(value_max):
                x0.append(elembis)  #amplitude
                x0.append(1) #sigma
                x0.append(0.1) #skewness



        print "---------------------------"
        
	def residuals(coeffs, y, time):
            return fit(time, coeffs) -y


        res = least_squares(residuals, x0,f_scale=0.01, loss = "cauchy", args=(elem, time)) #function from scipy
        x = res.x


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

        #Selection of the peaks that are far enough from the beginning or the end of the curve to be taken in account. A peak that is close to the borders of the curve will not be fitted correctly.
        amplitude = []
        std_dev = []
        width = 2 # Arbitrary, it determines how far a peak must be from the beginning or end of the curve in order to take it in account in the final result. Here, a peak must be 2 times its own width away from the borders.
	
	
	y=[]  
        z = []

        for i, elembis in enumerate(pos_max):
            if ((elembis - width*std_dev_raw[i]/day_per_pxl) >0) and ((elembis + width*std_dev_raw[i]/day_per_pxl) < len(elem)):

		amplitude.append(value_max[i])
                std_dev.append(std_dev_raw[i])
                y.append(elem[int(elembis - width * std_dev_raw[i] / day_per_pxl)])
                y.append(elem[int(elembis + width * std_dev_raw[i] / day_per_pxl)])
                z.append(time[elem.index(elem[int(elembis - width * std_dev_raw[i] / day_per_pxl)])])
                z.append(time[elem.index(elem[int(elembis + width * std_dev_raw[i] / day_per_pxl)])])
            else:
                print std_dev_raw[i]
                print "Unable to determine the amplitude"

	if not data:		
		j= array_index(LC,elem,reverse)
		print "ppppppppppppppppppppppppppppppppppppppppppppppppppp"
		#j= array_index(LC,elem.tolist(),reverse) #index of the treated lightcurve in the dictionnary previously started.
		print j
		
		
		
		
        if showplot:
            if not data:
                #arrow= [database["trajectory %s"%(j+1)][0], database["trajectory %s"%(j+1)][1], int(database["trajectory %s"%(j+1)][0] + database["trajectory %s"%(j+1)][2]*n_step), int(database["trajectory %s"%(j+1)][1]+ database["trajectory %s"%(j+1)][3]*n_step)]
		arrow= [database["trajectory %s"%(j)][0], database["trajectory %s"%(j)][1], int(database["trajectory %s"%(j)][0] + database["trajectory %s"%(j)][2]*n_step), int(database["trajectory %s"%(j)][1]+ database["trajectory %s"%(j)][3]*n_step)]
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
                    ax.plot(time, elem, label = "Simulated microlensing event")
                    ax.plot(time, fit(time, res.x), label = "Skewed-gaussian fit")
                    ax.plot(time_max, value_max, "o")
                    if len(z) > 0:
                        ax.plot(z, y , "o")

                    print np.min(elem)

                    ax.set_ylim((np.max(elem), 1.2*np.min(elem)))                    
                else:
		
                    ax.plot(time, 0*np.ones(len(elem))-elem, label = "Simulated microlensing event")
                    ax.plot(time, 0*np.ones(len(fit(time, res.x)))-fit(time, res.x), label = "Skewed-gaussian fit")
                    ax.plot(time_max, 0*np.ones(len(value_max))-value_max, "o")
                    if len(z)>0:
                        ax.plot(z,0*np.ones(len(y))-y,"o")
                    ax.set_ylim((-np.min(elem), -1.2*np.max(elem)))


                os.chdir(workdir+"results")
                plt.show()
                fig.savefig("ex_LC_%s.png" %(j+1) )

            else :
                if reverse:
                    plt.plot(time, elem, label = "PyCS estimation of the microlensing behavior")
                    plt.plot(time, fit(time, x), label = "Skewed-gaussian fit")
                    plt.plot(time_max, 0*np.ones(len(value_max))-value_max, "o")
                    if len(z) > 0:
                        plt.plot(z, 0*np.ones(len(y))-y, "o")
                    #plt.ylim((0, 1.5*np.max(elem)))


                else:
                    plt.plot(time, elem, label = "PyCS estimation of the microlensing behavior")
                    plt.plot(time, fit(time, x), label = "Skewed-gaussian fit")
                    plt.plot(time_max, value_max, "o")
                    if len(z)>0:
                        plt.plot(z,y,"o")
# plt.ylim((0, -(np.max(elem)+np.max(elem)/2)))

                plt.legend(fontsize = 20)
                plt.xlabel("Time [days]")
                plt.ylabel("Relative magnitude")

                os.chdir(workdir+"results")
                plt.show()
                plt.savefig("data_%s_%s.png" % (reverse, model))



        if data: 
            if reverse:

                return 0*np.ones(len(amplitude))-amplitude, std_dev
            else :
                return amplitude, std_dev

        else:
            if reverse: 
		f_amp = 0*np.ones(len(amplitude))-amplitude
		print type(f_amp.tolist())

            else :
	    	f_amp = amplitude

            database["trajectory " + str(j)] = database["trajectory " + str(j)] + [ pos_max, std_dev]

    else:
        print "No interesting peaks here"
	



def compare2data(database, amplitude, std_dev, error_margin):


# This function was created to calculate the number of simulations matching the experimental data but it is not optimal and not that useful, forget it ...

    #lc and time are given by the draw_LC function
    #model : can be either voigt or gaussian for now
    #data : data to which you wanna compare the simulations
    #error_margin : first on the amplitude of the signal and then on the width


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

                            comparison[key] = elem
                            temp.append(j)
                            temp.append(i)
                            count_maybe+=1
                        else :

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
                        count_slow += 1
                        comparison[key].append("Amplitude match for peak %s in sim with peak %s in data" % (elembis[2*i+1], elembis[2*i]))
                        comparison[key].append("Too fast : sig_sim vs sig_data : %s vs %s" % (elem[6][elembis[2*i+1]], std_dev[elembis[2*i]]))
			
                    elif elem[6][elembis[2*i+1]] >= std_dev[elembis[2*i]] + error_margin[1]:
                        count_fast += 1
                        comparison[key].append("Amplitude match for peak %s in sim with peak %s in data" % (elembis[2*i+1], elembis[2*i]))
                        comparison[key].append("Too slow : sig_sim vs sig_data : %s vs %s" % (elem[6][elembis[2*i+1]], std_dev[elembis[2*i]]))




    return comparison, match, count_maybe, count_winner, count_looser, count_slow, count_fast

def histo(db, filename):

# Collects the amplitudes (amp) and width of the peaks (duration) in the database of simulated curves to plot them

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
                    total.append(temp) #Data put in the form of a list of [amp duration] necessary for the type of plot I made here
		    
		    
                    amp.append(elembis)   # In case you want to plot the data differently amp regroups every amplitude found and duration every width.
                    duration.append(elem[6][i])

    total = np.asarray(total)
 
    toplot=np.column_stack((amp, duration))
    figure = corner.corner(toplot, labels=[r"$Duration$", r"$Amplitude$"], show_titles=True)
      
    fig.savefig(filename)


#######################################  MAIN  #################################################


#Example of how to run the code !


map = '../data/convoluted_map_A_fft_thin_disk_30.fits'
img = fits.open(map, mode='update')[0]
map_conv = img.data[:, :] 


database = {}

# create coordinates of starting points and velocities
v_x_int= 1
v_y_int = 0
n_step = 700

sampling = np.arange(0, 8000, 1000)
x_start= np.repeat(sampling, len(sampling))
y_start = np.tile(sampling, len(sampling))
v_x = v_x_int*np.ones(len(x_start))
v_y = v_y_int*np.ones(len(y_start))


# drawing the curves
LC,time = draw_LC(x_start, y_start, v_x, v_y, n_step, map_conv)
print len(LC)
print type(LC[1])

delta = 0.03

#fit the curves using multiprocessing
#peak_fitting(LC[10], time, delta,map_conv, showplot=True,model = "skewed_gaussian", reverse = False)


n_cpu = multiprocessing.cpu_count()

pool = multiprocessing.Pool(int(n_cpu))
#pool = multiprocessing.Pool(1)
parallel_peak_fitting = partial(peak_fitting,time = time, delta = delta, showplot = True, model="skewed_gaussian", reverse = True)
pool.map(parallel_peak_fitting, LC)


#fit the data
#amplitude, std_dev = peak_fitting(ml_A[0], jds, delta,delta2=delta, data = True, model = "skewed_gaussian2", showplot=False, reverse=True)

#visualize the results !
histo(database, "demo.png")
