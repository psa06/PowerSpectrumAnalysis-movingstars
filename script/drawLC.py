import numpy as np
from astropy.io import fits
import os,sys
from scipy.optimize import least_squares
import pickle as pkl
#from sklearn.cluster import KMeans
import multiprocessing
from functools import partial
import corner
import matplotlib.pyplot as plt

execfile("useful_functions.py")


font = {'family' : 'normal',
        'size'   : 22}

plt.rc('font', **font)

einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = cm_per_pxl/(30000000000*3600*24)
print ld_per_pxl
print ld_per_pxl*30
print cm_per_pxl*30
print ld_per_pxl*500
print cm_per_pxl*500
v_source = 7000 #km/s
day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)
print day_per_pxl
def draw_LC(x_start, y_start, v_x, v_y, n_points, map):
#    x_start,y_start : Lists of integers. Lists of all the starting coordinates you want to test . Pixel
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
#        lc.append(-2.5*np.log(temp)) #for A

#        temp = 0*np.ones(len(temp))-temp #for B
        lc.append(temp) # +8.2*np.ones(len(temp)) for the test map


        database["trajectory %s"%(i)] = [x_start[i], y_start[i], v_x[i], v_y[i]]

    if len(path_x) == 0:
        sys.exit("No lightcurve to analyse")


    return lc, time




def peak_fitting(elem, time, delta,delta2, model="gaussian", data = False, showplot = False, reverse = False):
    #lc and time are given by the draw_LC function
    #delta : float : necessary for the use of peakdet_org function
    #model : can be either voigt or gaussian or lorentzian for now
    #showplot : Show individually the fit on each lightcurve simulated ... or not
    
    result_f = []
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
                x0.append(0.1) #sigma
                x0.append(0.5) #skewness


        print "---------------------------"
        def residuals(coeffs, y, time):
            return fit(time, coeffs) -y


        res = least_squares(residuals, x0,f_scale=0.1, loss = "cauchy", args=(elem, time))
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
        width = 1.5
        if data:
            width = 2
        security = 600
        for i, elembis in enumerate(pos_real_max):

            if (std_dev_raw[i] >0) and (std_dev_raw[i]<security) and ((elembis - width*std_dev_raw[i]/day_per_pxl) >0) and ((elembis + width*std_dev_raw[i]/day_per_pxl) < len(elem)):
                print "+++++++++++++++++++++"
                print std_dev_raw
                print value_real_max
                amp = elem[elembis] - (elem[int(elembis - width*std_dev_raw[i]/day_per_pxl)]+elem[int(elembis + width*std_dev_raw[i]/day_per_pxl)])/2
                if amp > delta2:
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
                    ax.plot(time, -elem, label = "Simulated microlensing event")
                    ax.plot(time, -fit(time, res.x), label = "Skewed-gaussian fit")
                    ax.plot(time_real_max, 0*np.ones(len(value_real_max))-value_real_max, "o")
                    if len(z) > 0:
                        ax.plot(z, 0*np.ones(len(y))-y , "o")

                    print np.min(elem)

                    ax.set_ylim((0, -1.5*np.max(elem)))
                else:
                    ax.plot(time, elem, label = "raw_data")
                    ax.plot(time, fit(time, res.x), label = "Skewed-gaussian fit")
                    ax.plot(time_real_max, value_real_max, "o")
                    if len(z)>0:
                        ax.plot(z,y,"o")
                    ax.set_ylim((0, -(np.max(elem)+np.max(elem)/2)))


                os.chdir("/home/epaic/Documents/Astro/TPIVb/results")
                plt.show()
                fig.savefig("ex_LC_%s.png" %(j+1) )

            else :
                if reverse:
                    plt.plot(time, -elem, label = "PyCS estimation of the microlensing behavior")
                    plt.plot(time, -fit(time, x), label = "Skewed-gaussian fit")
                    plt.plot(time_real_max, 0*np.ones(len(value_real_max))-value_real_max, "o")
                    if len(z) > 0:
                        plt.plot(z, 0*np.ones(len(y))-y, "o")
                    plt.ylim((0, -1.25*np.max(elem)))


                else:
                    plt.plot(time, elem, label = "PyCS estimation of the microlensing behavior")
                    plt.plot(time, fit(time, x), label = "Skewed-gaussian fit")
                    plt.plot(time_real_max, value_real_max, "o")
                    if len(z)>0:
                        plt.plot(z,y,"o")
# plt.ylim((0, -(np.max(elem)+np.max(elem)/2)))

                plt.legend(fontsize = 20)
                plt.xlabel("Time [days]")
                plt.ylabel("Relative magnitude")

                os.chdir("/home/epaic/Documents/Astro/TPIVb/results")
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
                f_amp = 0*np.ones(len(amplitude))-amplitude

            else :
                f_amp = amplitude
            result_f.append(["trajectory %s"%(j)] +database["trajectory " + str(j)] + [ f_amp.tolist(),pos_real_max, std_dev])

	    return result_f
		
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

def histo(db, v_x_int, v_y_int, amplitude=None, std_dev=None):

    for i,elembis in enumerate(db):
        amp = []
        duration = []
        total = []
        for key,elem in enumerate(elembis):
    #        print "1111111111111111111111111111"
    #	print elem[0]
            if len(elem[0][5]) >0:
                for i,elembis in enumerate(elem[0][5]):
    #	    	print "222222222222222222"
    #		print elembis
                    if elem[0][7][i] >10:
                        temp = []

                        temp.append(20*elembis)
                        temp.append(elem[0][7][i])

                        total.append(temp)
                        amp.append(np.abs(elembis))
                        duration.append(elem[0][7][i])

#    print amp
#    print duration
    total = np.asarray(total)
    ##############################################################################
    # Compute clustering with Means
#    n_clus = 3
#    k_means = KMeans(n_clusters= n_clus)
#    k_means.fit(total)
#    centers = k_means.cluster_centers_
#    y_kmeans = k_means.predict(total)

    ##############################################################################
    # Plot result

#    colors = ['#4EACC5', '#FF9C34', '#4E9A06']

#    fig = plt.figure(figsize=(6, 6))
#    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
#    main_ax = fig.add_subplot(grid[:-1, 1:])
#    y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
#    x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)
#    # scatter points on the main axes
#    main_ax.scatter(duration, amp , c = y_kmeans, cmap='viridis')
#    main_ax.scatter(centers[:, 1], centers[:, 0] / 20, c='black', s=200, alpha=0.5)
#    main_ax.scatter(std_dev, 0 * np.ones(len(std_dev)))
#    main_ax.scatter(0 * np.ones(len(amplitude)), amplitude)
#    main_ax.set_xlim((0,150))
#    main_ax.set_ylim((-2,2))
#    main_ax.set_xlabel("Duration (standard deviation of the peaks)")
#    main_ax.set_ylabel("Amplitude")
#    # histogram on the attached axes
#    x_hist.hist(duration, 40, histtype='stepfilled',
#                orientation='vertical')
#    x_hist.set_xlabel("Duration (days)")
#    x_hist.invert_yaxis()

#    y_hist.hist(amp, 40, histtype='stepfilled',
#                orientation='horizontal')
#    y_hist.set_ylabel("Amplitude")
#    y_hist.invert_xaxis()

#    plt.show()



    toplot=np.column_stack((duration, amp))
#    fig = corner.corner(toplot, labels=[r"$Duration$", r"$Amplitude$"], show_titles=True)


    value1 = np.abs(amplitude)
    value2 = std_dev
    print "mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm"
    print value1
    print value2
    ndim =2
    # Make the base corner plot
    figure = corner.corner(toplot, labels=[r"$Duration [days]$", r"$Amplitude [mag]$"], show_titles=True, hist_kwargs={} )

    # Extract the axes
    axes = np.array(figure.axes).reshape((ndim, ndim))

    color = ["g", "r"]
    # Loop over the diagonal
    for i in range(ndim):
        ax = axes[i, i]
        for j,elem in enumerate(value1):
            ax.axvline(elem, color=color[j])
            ax.axvline(value2[j], color=color[j])

    # Loop over the histograms
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            for j,elem in enumerate(value1):
                ax.axvline(elem, color=color[j])
                ax.axvline(value2[j], color=color[j])
                ax.axhline(elem, color=color[j])
                ax.axhline(value2[j], color=color[j])

    plt.show()
    os.chdir('/home/epaic/Documents/Astro/TPIVb/results')

    figure.savefig("histo_A_ampVSduration_%s_%s_%s_30.png" % (v_x_int, v_y_int, n_step))


    # best fit of data
 #   (mu, sigma) = skewnorm.fit(amp)

    # the histogram of the data
#    n, bins, patches = plt.hist(amp, 60, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
#    from lmfit.models import SkewedGaussianModel
#    model = SkewedGaussianModel()

    # set initial parameter values#
#    params = model.make_params(amplitude=1, center=np.median(amp), sigma=1, gamma=0)

    # adjust parameters  to best fit data.
#    result = model.fit(amp, params, x=duration)
#    y = plt.plot(bins, result.best_fit)
#    l = plt.plot(bins, y, 'r--', linewidth=2)

    # plot
#    plt.xlabel('Amplitude (mag)')
#    plt.ylabel('Count')
    #plt.title(r'$\mathrm{Histogram\ of\ Amplitude:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
 #   plt.grid(True)

#    plt.show()

 #   fig.savefig("histo_A_amp_%s_%s_%s_30.png" % (v_x_int, v_y_int, n_step))

#    (mu, sigma) = norm.fit(duration)

    # the histogram of the data
 #   n, bins, patches = plt.hist(duration, 60, normed=1, facecolor='green', alpha=0.75)

    # add a 'best fit' line
#    y = mlab.normpdf(bins, mu, sigma)
 #   l = plt.plot(bins, y, 'r--', linewidth=2)

    # plot
##    plt.xlabel('Duration (Days)')
  #  plt.ylabel('Count')
    #plt.title(r'$\mathrm{Histogram\ of\ Duration:}\ \mu=%.3f,\ \sigma=%.3f$' % (mu, sigma))
 #   plt.grid(True)

  #  plt.show()

#    fig.savefig("histo_A_duration_%s_%s_%s_30.png" % (v_x_int, v_y_int, n_step))

#    fig, axes = plt.subplots(1,2,figsize = [14,6])
#    ax=axes[0]
#    ax.scatter(duration, amp , c = y_kmeans, cmap='viridis')
#    ax.scatter(centers[:, 1], centers[:, 0]/20, c='black', s=200, alpha=0.5);
    #for k, col in zip(range(n_clus), colors):
    #    my_members = k_means_labels == k
    #    cluster_center = k_means_cluster_centers[k]
    #    plt.plot(total[my_members][0], total[my_members][1], 'w',
    #             markerfacecolor=col, marker='.')
    #    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #             markeredgecolor='k', markersize=6)
#    ax.set_xlabel("Duration (standard deviation of the peaks)")
#    ax.set_ylabel("Amplitude")

#    ax = axes[1]
#    ax.hist(amp, bins = 35)
#    ax.plot(amplitude, 0*np.ones(len(amplitude)))
#    ax.set_xlabel("Amplitude")

#    ax = axes[2]
#    ax.hist(std_dev, bins = 35)
#    ax.plot(std_dev, 0*np.ones(len(std_dev)))
#    ax.set_xlabel("Duration")

#    plt.show()
#    fig.savefig("histo_A_ampVSduration_%s_%s_%s_500.png"%(v_x_int, v_y_int,n_step))



 #   plt.plot(duration, amp , "o")
 #   plt.savefig("histo_ampVSduration_smooth_%s_%s.png"%(v_x_int, v_y_int))
 #   plt.show()


#######################################  MAIN  #################################################



os.chdir('/home/epaic/Documents/Astro/TPIVb/data')
map = '../mapA.fits'
#map = '/home/epaic/Documents/Astro/TPIVb/data/convoluted_map_A_fft_thin_disk_500.fits'

#map = 'map_convolved_25.fits'
img = fits.open(map, mode='update')[0]
map_conv = img.data[:, :]

#display_map(map)
#plt.savefig("map_with_source.png")
#plt.show()
#exit()

[jds, ml_A, ml_B] = pkl.load(open('ml.pkl', 'rb'))

ml_A = [ml_A]
ml_B = [ml_B]


#[jds, intensity] = pkl.load(open('artificial_peak.pkl', 'rb'))
#intensity = [intensity]

database = {}
result = []
v_x_int= 1
v_y_int = 1
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


parallel_peak_fitting = partial(peak_fitting,time = time, delta = delta, delta2 = delta, showplot = True, model="skewed_gaussian2", reverse = True)

#peak_fitting(LC, time, delta, delta2=delta, showplot=True,model = "skewed_gaussian2", reverse = True)
#peak_fitting(LC, time, delta, delta2=delta, showplot=False,model = "skewed_gaussian2", reverse = False)

pool = multiprocessing.Pool(n_cpu)
res = pool.map(parallel_peak_fitting, LC)


print "DATABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAASE"

#result_f =[]
#for i,elem in enumerate(res):
#    if  not elem is None:
#     	result_f.append(elem)


os.chdir('/home/epaic/Documents/Astro/TPIVb/results')

#with open('result_%s_%s_%s.pkl'%(v_x_int,v_y_int,n_step), 'wb') as handle:
#    pkl.dump([result_f], handle, protocol=pkl.HIGHEST_PROTOCOL)

#print result_f

#[result1] = pkl.load(open('result_1_1_700.pkl', 'rb'))
#[result2] = pkl.load(open('result_1_0_700.pkl', 'rb'))

#result = [result1]
#error_margin= [0.005,3]


#amplitude, std_dev = peak_fitting(ml_A[0], jds, delta,delta2=delta, data = True, model = "skewed_gaussian2", showplot=True, reverse=True)
#histo(result, v_x_int, v_y_int, amplitude, std_dev)
#histo(result, v_x_int, v_y_int)

#amplitude2, std_dev2 = peak_fitting(ml_A, jds, delta,delta2=delta, data = True, model = "skewed_gaussian2", showplot=True, reverse=False)
#histo(database, v_x_int, v_y_int, amplitude2, std_dev2)

#comp, match, maybe, winner, looser, slower, faster = compare2data(result_f, amplitude, std_dev, error_margin )
#comp2, match2, maybe2, winner2, looser2, slower2, faster2 = compare2data(database, amplitude2, std_dev2, error_margin )


os.chdir('/home/epaic/Documents/Astro/TPIVb/results')


#file = open("result.txt", 'a')
#file.write(str(v_x_int) + "\t" + str(v_y_int)+ "\t" + str(delta)+ "\t" + str(error_margin[0]) + "\t" + str(error_margin[1])+ "\t" + str(looser) + "\t" + str(maybe) + "\t" + str(winner)+ "\n")
#file.close()


'''
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

'''
