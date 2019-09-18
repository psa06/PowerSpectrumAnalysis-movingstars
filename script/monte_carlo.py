
from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import os,sys
import timeit
import multiprocessing
from functools import partial
import pickle as pkl
import pycs
import astropy
import astroML.time_series as amlt
import matplotlib

storagedir = "/run/media/epaic/TOSHIBA EXT/maps/Q0158/"

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)

execfile("useful_functions.py")
#np.seterr(divide='ignore', invalid='ignore')
database = []

einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = cm_per_pxl/(30000000000*3600*24)

v_source = 100 #km/s
day_per_pxl = cm_per_pxl/(100000*v_source*3600*24)
v_source_in_map = 1/day_per_pxl
nslopes=10
totlength = 0




def draw_LC(n_curves, slope, map, lengthmin):
    #    x_start,y_start : Lists of integers. Lists of all the starting coordinates you want to test
    #   v_x, v_y : Lists of integers. Lists of all the coordinate of the velocity vector of the source on the lense plane you want to test (for now in pxl/point)
    #   n_points : Integer. Length of LC you want to simulate.
    #   map : Array of the convoluted map

    # One element of x_start, y_start, v_x and v_y is used to create a single lightcurve. If those parameters are lists instead of integers you will get list of lightcurves stored in lc. time is a single list valid for every lightcurve.
    scan = np.linspace(0, len(map), n_curves).tolist()
    step = 1
    database_length = len(database)

#    if len(x_start) != len(y_start):
#        print "--------------------SHAME ! SHAME ! SHAME !--------------------------"
#        print "Your input vectors don't have the same length"
#        print "x_start : " + str(len(x_start)) + " y_start : " + str(len(y_start))
#        sys.exit()


    path_x = []
    path_y = []
    lc = []

    temp = np.arange(0, len(map) - 1, step).tolist()
    rev_temp = np.arange(0,len(map) - 1, step).tolist()
    rev_temp.reverse()
   # print "aaaaaaaaaaaaaa"
   # print temp
   # print rev_temp
    for i,elem in enumerate(scan):
        ii=0
        temp2=[]

        while  0<= slope*temp[ii] + scan[i] <len(map)-1 :
            temp2.append(int(slope*temp[ii] + scan[i]))
            ii+=1
            if ii >= len(temp)-1:
                break

        if len(temp2) > lengthmin:
#            path_x.append(temp[:len(temp2)])
#            path_y.append(temp2)
#            path_x.append(rev_temp[:len(temp2)])
#            path_y.append(temp2)

#            path_y.append(temp[:len(temp2)])
#            path_x.append(temp2)
#            path_y.append(rev_temp[:len(temp2)])
#            path_x.append(temp2)


            lc.append(np.multiply(-2.5,np.log10(map[temp[:len(temp2)],temp2])))  # -2.5 log() to convert flux into mag
            lc.append(np.multiply(-2.5,np.log10(map[rev_temp[:len(temp2)], temp2])))
            lc.append(np.multiply(-2.5,np.log10(map[temp2, temp[:len(temp2)]])))
            lc.append(np.multiply(-2.5,np.log10(map[temp2, rev_temp[:len(temp2)]])))

            database.append([temp[:len(temp2)][0], temp2[0], temp[:len(temp2)][-1], temp2[-1]])
            database.append([rev_temp[:len(temp2)][0], temp2[0], rev_temp[:len(temp2)][-1], temp2[-1]])
            database.append([temp2[0], temp[:len(temp2)][0], temp2[-1], temp[:len(temp2)][-1]])
            database.append([temp2[0], rev_temp[:len(temp2)][0], temp2[-1], rev_temp[:len(temp2)][-1]])

    return lc


def new_draw_LC(param, data, errdata,map, time, chi_max):

    x_start = param[0]
    y_start = param[1]
    v = param[2]
    angle = param[3]

    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)

    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)



    if x_start + (time[-1] - time[0]) * v_x <= len(map) and y_start + (time[-1] - time[0]) * v_y <= len(map) and x_start + (time[-1] - time[0]) * v_x >= 0 and y_start + (time[-1] - time[0]) * v_y >= 0:

        if v_x == 0:
            path_x = x_start * np.ones(len(time))
        else:
            path_x = np.add(np.multiply(np.add(time, -time[0]),v_x), x_start)
        if v_y == 0:
            path_y = y_start * np.ones(len(mjhd))
        else:
            path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

        path_x= path_x.astype(int)
        path_y = path_y.astype(int)
        temp = np.multiply(-2.5, np.log10(map[path_y, path_x]))  # -2.5 log() to convert flux into mag

        lc = temp - temp[0] * np.ones(len(temp))
        result = chi2(lc, data, errdata)
#        database.append([path_x[0], path_y[0], path_x[-1], path_y[-1]])
        if result < chi_max:
#            database.append([path_x[0],path_y[0], path_x[-1], path_y[-1]])
            return lc, result, x_start, y_start, v, angle

def residuals_LC(params, errdata, map, time):
    x_start = params[0]
    y_start = params[1]
    v = params[2]
    angle = params[3]

    v_x = np.multiply(v, np.cos(angle))
    v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)

    v_y = np.multiply(v, np.sin(angle))
    v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)

    if x_start + (time[-1] - time[0]) * v_x <= len(map) and y_start + (time[-1] - time[0]) * v_y <= len(map) and x_start + (time[-1] - time[0]) * v_x >= 0 and y_start + (time[-1] - time[0]) * v_y >= 0:
        if v_x == 0:
            path_x = x_start * np.ones(len(time))
        else:
            path_x = np.add(np.multiply(np.add(time, -time[0]),v_x), x_start)
        if v_y == 0:
            path_y = y_start * np.ones(len(mjhd))
        else:
            path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

        path_x= path_x.astype(int)
        path_y = path_y.astype(int)
        temp = np.multiply(-2.5, np.log10(map[path_y, path_x]))  # -2.5 log() to convert flux into mag

        err_mag =np.random.normal(np.mean(errdata), np.var(errdata), len(temp))
        result = residuals(time, temp, err_mag)
        lc = temp - temp[0] * np.ones(len(temp))
        return lc, result, params

def residuals(time, mag, magerrs):
    lc = pycs.gen.lc.factory(time, mag, magerrs=magerrs)
    spline=pycs.gen.spl.fit([lc], knotstep =600, bokeps=400,verbose=False)
    pycs.sim.draw.saveresiduals([lc],spline)
    #pycs.gen.lc.display([lc], [spline])

    return np.sum(np.divide(np.abs(np.array(lc.residuals)),magerrs))


def sim_LC(x_start, y_start, v_x, v_y, time, err_data, map):
    #    x_start,y_start : Lists of integers. Lists of all the starting coordinates you want to test
    #   v_x, v_y : Lists of integers. Lists of all the coordinate of the velocity vector of the source on the lense plane you want to test (for now in pxl/point)
    #   n_points : Integer. Length of LC you want to simulate.
    #   map : Array of the convoluted map

    # One element of x_start, y_start, v_x and v_y is used to create a single lightcurve. If those parameters are lists instead of integers you will get list of lightcurves stored in lc. time is a single list valid for every lightcurve.

    if x_start + (time[-1] - time[0]) * v_x <= len(map) and y_start + (time[-1] - time[0]) * v_y <= len(map) and x_start + (time[-1] - time[0]) * v_x >= 0 and y_start + (time[-1] - time[0]) * v_y >= 0:
        if v_x == 0:
            path_x = x_start * np.ones(len(time))
        else:
            path_x = np.add(np.multiply(np.add(time, -time[0]),v_x), x_start)
        if v_y == 0:
            path_y = y_start * np.ones(len(mjhd))
        else:
            path_y = np.add(np.multiply(np.add(time, -time[0]), v_y), y_start)

        path_x= path_x.astype(int)
        path_y = path_y.astype(int)
        temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),np.random.normal(0, np.mean(err_data),len(path_y)))  # -2.5 log() to convert flux into mag

        lc = temp - temp[0] * np.ones(len(temp))
        database.append([path_x[0],path_y[0], path_x[-1], path_y[-1]])
        return lc, path_x, path_y



def chi2(sim,data,errdata):
    chi2 = np.sum(np.power(np.divide(sim-data, errdata),2))
    return chi2/len(sim)

def compare(v, sim, data, errdata, time):
    #v en pxl par jour
    
    length_subsim = int((time[-1]-time[0])*v)+2
    subset=[]
    step = 20
    pxl_parc = np.multiply(np.substract(time, time[0]),v)
    pxl_parc = pxl_parc.astype(int)
    #for i, elem in enumerate(time):
    #    pxl_parc.append(int((time[i] - time[0]) * v))

    new_subset = []
    for j,lc in enumerate(sim):
        temp_res = []
        for i in range(int((len(lc)-1-length_subsim)/step)):
            temp = np.array(lc[i*step:length_subsim +i*step-1])
            temp2 = temp[pxl_parc]-temp[0]*np.ones(len(temp[pxl_parc]))
            new_subset.append(temp2)
            temp_res.append(chi2(temp2, data,errdata))

    return [min(temp_res), new_subset[temp_res.index(min(temp_res))],v]


def find_min(res):
    result = []
    sublc = []
    v = []
    for i, elem in enumerate(res):
        if len(elem)>0:
            result.append(elem[0])
            sublc.append(elem[1])
            v.append(elem[2])

    return min(result), sublc[result.index(min(result))], v[result.index(min(result))], result, v


#----------------------------------Import Data-----------------------------------------------
f = open(datadir+"/microlensing/data/J0158_Euler_microlensing_upsampled_B-A.rdb","r")
#f = open(datadir+"/microlensing/data/RXJ1131_ALL_microlensing_upsampled_B-A.rdb","r")

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


#mjhd = mjhd[np.where(mjhd<54250 )]
#mag_ml = mag_ml[np.where(mjhd<54250 )]
#err_mag_ml = err_mag_ml[np.where(mjhd<54250)]

#mjhd = mjhd[np.where(mjhd>53800 )]
#mag_ml = mag_ml[np.where(mjhd>53800)]
#err_mag_ml = err_mag_ml[np.where(mjhd>53800)]


#res_ref = residuals(mjhd,mag_ml, err_mag_ml)
#print "++++++++++++++++++++++++++++++"
#print res_ref

lengthmin = (mjhd[-1]-mjhd[0])*v_source_in_map

#map_08 = datadir+"convolved_map_A-B_fft_thin_disk_30_fml09.fits"
#img_08 = fits.open(map_08)[0]
#final_map_08 = img_08.data[:, :]

map = datadir+"Q0158/FML0.9/mapA-B_fml09_R20.fits"
img = astropy.io.fits.open(map)[0]
final_map = img.data[:, :]

#----------------------------------Create Sim---------------------------------------------------

i =0
while i <10000:
    #x0 = np.random.random_integers(2400, 3400)
    #y0 = np.random.random_integers(700, 1600)
    x0 =2970
    y0=len(final_map)-1-2792
    i+=1
    #theta = np.random.uniform(0,2*np.pi)
    #theta = 0.222173554445
    #v_x_sim = np.multiply(v_sim,np.cos(theta))
    #v_x_sim = np.divide(np.multiply(100000*3600*24,v_x_sim),cm_per_pxl)
    v_x_sim = 0.046856
    #v_x_sim = 0
    #v_y_sim = np.multiply(v_sim,np.sin(theta))
    #v_y_sim = np.divide(np.multiply(100000*3600*24,v_y_sim),cm_per_pxl)
    #v_y_sim = 0.08381402
    v_y_sim = -0.05389616
    if x0 + (mjhd[-1] - mjhd[0]) * v_x_sim <= len(map) and y0 + (mjhd[-1] - mjhd[0]) * v_y_sim <= len(map) and x0 + (mjhd[-1] - mjhd[0]) * v_x_sim >= 0 and y0 + (mjhd[-1] - mjhd[0]) * v_y_sim >= 0:
        break

v_sim = np.sqrt(v_x_sim**2+v_y_sim**2)*cm_per_pxl/(100000*3600*24)
print v_sim
sys.exit()
lc_sim,path_x, path_y = sim_LC(x0,y0, v_x_sim, v_y_sim, mjhd,err_mag_ml, final_map)

#arrows = []
#for i,trajectory in enumerate(database):
#    temp = [trajectory[0], trajectory[1], trajectory[2], trajectory[3]]
#    arrows.append(temp)

#display_multiple_trajectory(arrows, map)

#plt.plot(mjhd, lc_sim, "o")
#plt.ylim([max(lc_sim), min(lc_sim)])
#plt.show()

#arrow = [ len(final_map)-path_y[0],path_x[0], len(final_map)-path_y[-1], path_x[-1]]
arrow = [ path_y[0],path_x[0], path_y[-1], path_x[-1]]
#arrow = [ path_x[0], len(final_map)-path_y[0], path_x[-1],len(final_map)-path_y[-1]]
#arrow = [ path_x[0],path_y[0], path_x[-1], path_y[-1]]
center = [int((arrow[0] + arrow[2]) / 2), int((arrow[1] + arrow[3]) / 2)]
print center
print
crop_size = 500
crop = [np.max([center[0] - crop_size, 0]), np.min([center[0] + crop_size, len(final_map)]),
        np.max([center[1] - crop_size, 0]), np.min([center[1] + crop_size, len(final_map)])]

print arrow
print center
print crop

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
display_trajectory(crop, arrow, axes, map=map)
ax = axes[0]

ax.plot(mjhd, lc_sim, 'o')
ax.set(xlabel='Time [days]', ylabel='Magnitude')
axes[1].set(title = 'Zoom of the map')
#ax.plot(mjhd, 0 * np.ones(len(fit(time, res.x))) - fit(time, res.x), label="Skewed-gaussian fit")
#ax.plot(mjhd, mag_ml, "o", label='Micro lensing curve')
#ax.set_ylim((-np.min(elem), -1.2 * np.max(elem)))

#os.chdir(workdir + "results")
plt.show()
#fig.savefig("ex_LC_%s.png" % (j + 1))
sys.exit()

#--------------------------------First Step-----------------------------------

n_times = 20
result = []

for ii in np.arange(n_times):
    n_traj = 100000

    start2 = timeit.default_timer()
    #v = np.abs(np.random.normal(500,200,n_traj))
    x_start = np.random.random_integers(0, len(map) - 1,n_traj)
    y_start = np.random.random_integers(0, len(map) - 1,n_traj)
    angle = np.random.uniform(0, 2 * np.pi,n_traj)
    v = np.random.uniform(100,1500,n_traj).tolist()

    params = []
    for i,elem in enumerate(v):
        params.append([x_start[i], y_start[i],elem, angle[i]])

    #v = 1000*np.ones(n_traj)

    #map =np.tile([,int(n_traj/4))








    #result = []
    #for i,elem in enumerate(params):
    #    temp = residuals_LC(elem, errdata = err_mag_ml, map = final_map, time = mjhd)
    #    if temp is not None:
    #        result.append(temp[1])
    #        lc.append(temp[0])

    #max_chi = max(result)
    #best_lc = lc[result.index(max_chi)]

    #fig, axes = plt.subplots(1, 2, figsize=(14,6))
    #ax1 = axes[0]
    #ax3 = axes[1]

    #ax1.hist(result,density = True, color="r", bins= 100)
    #ax1.plot([res_ref, res_ref],[0, ax1.get_ylim()[1]], "k")

    #ax1.legend(loc='upper right')

    #ax3.plot(mjhd, best_lc , "+", label ="Most variable curve")
    #ax3.legend()
    #ax3.set_ylim(1.2*max(best_lc), 1.2*min(best_lc))

    #plt.show()
    #fig.savefig(resultdir + "look4variability_%s_1.png" % (n_traj))



    #n_cpu = multiprocessing.cpu_count()
    #print n_cpu
    #pool = multiprocessing.Pool(2)

    #parallel_residuals_LC = partial(residuals_LC, errdata = err_mag_ml, map = final_map, time = mjhd)

    #result=pool.map(parallel_residuals_LC, params)

    #parallel_new_draw_LC = partial(new_draw_LC, data = lc_sim, errdata = err_mag_ml,map = final_map,chi_max =3000, time = mjhd)
    #result=pool.map(parallel_new_draw_LC, params)

    for param in params:
        temp = new_draw_LC(param,  data = lc_sim, errdata = err_mag_ml,map = final_map,chi_max =1000000, time = mjhd)
        if temp is not None:
            result.append(temp)





    stop3 = timeit.default_timer()
    print len(result)
    print stop3 -start2

result = np.array(result)
lc = result[:, 0]
chi = result[:, 1]
final_x = result[:, 2]
final_y = result[:, 3]
final_v = result[:, 4]
final_theta = result[:, 5]

result = []
params = []
print "First step finished"

    #-----------------------------------Second Step-------------------------------

for jj in np.arange(n_times):
    new_x = np.array([])
    new_y = np.array([])
    new_v = np.array([])
    new_theta = np.array([])

    n_supptraj = max([int(n_traj/len(final_v)), 1])
    pxl_freedom = 100
    for i,v in enumerate(final_v):
        #pxl_freedom = np.divide(np.multiply(100000 * 3600 * 24, param[2]), cm_per_pxl)*(mjhd[-1]-mjhd[0])/4
        #pxl_freedom = int(pxl_freedom)
        new_x = np.append(new_x, np.random.random_integers(max([final_x[i]-pxl_freedom, 0]),min([len(final_map)-1, final_x[i]+pxl_freedom]), n_supptraj))
        new_y = np.append(new_y, np.random.random_integers(max([final_y[i]-pxl_freedom, 0]),min([len(final_map)-1, final_y[i]+pxl_freedom]), n_supptraj))
        new_v = np.append(new_v, np.abs(np.random.uniform(final_v[i]-50,final_v[i]+50,n_supptraj)))
        new_theta = np.append(new_theta, np.random.uniform(final_theta[i]-np.pi/6,final_theta[i]+np.pi/6, n_supptraj))

    print "Second step in preparation"

    new_params = []
    for j,elem in enumerate(new_x):
        new_params.append([new_x[j], new_y[j], new_v[j], new_theta[j]])

    #parallel_new_draw_LC = partial(new_draw_LC, data = lc_sim, errdata = err_mag_ml,map = final_map,chi_max =3000, time = mjhd)

    #result2=pool.map(parallel_new_draw_LC, new_params)
    for param in new_params:
        temp = new_draw_LC(param,  data = lc_sim, errdata = err_mag_ml,map = final_map,chi_max =1000000, time = mjhd)
        if temp is not None:
            result.append(temp)


    #for i,res in enumerate(result2):
    #for i,elem in enumerate(params):
    #    res = new_draw_LC(elem, lc_sim, err_mag_ml, final_map, mjhd)
    #    if res is not None:
    #        lc=np.append(lc,res[0])
    #        chi=np.append(chi,res[1])
    #        final_x.append(res[2])
    #        final_y.append(res[3])
    #        final_v.append(res[4])
    #        final_theta.append(res[5])

    #result2 = filter(None, result2)
    stop5 = timeit.default_timer()
    print len(result)
    print len(chi)
    print stop5 - start2


result = np.array(result)
lc= np.append(lc,result[:,0])
chi = np.append(chi,result[:,1])
#final_params = np.append(final_params,result2[:,2])
final_v = np.append(final_v, result[:,4])




#----------------------Plot results---------------------------------------------



#weight = np.exp(-chi/2)
weight = []

for i,elem in enumerate(chi):
    temp = np.exp(-elem/2)
    weight.append(temp)

weight = np.array(weight)
print weight.shape
print final_v.shape
#weight = np.exp(-np.divide(np.power(chi,2),2))

#weighted_mean = np.average(final_v,weights=weight).astype(float)
#weighted_std = np.sqrt(np.average((final_v-weighted_mean)**2,weights=weight)).astype(float)

#print weighted_mean
#print weighted_std

percentiles = [0.15,0.5,0.82]
if len(final_v)>0:
#    quantiles = weighted_quantile(final_v, percentiles, sample_weight= weight)
#    print quantiles
    min_chi = min(chi)
    arg = np.where(chi==min_chi)[0][0]
    print arg
    best_lc = lc[arg]
    best_v = final_v[arg]

stop4 = timeit.default_timer()


print stop4 -start2
fig, axes = plt.subplots(1, 2, figsize=(20,9))
ax1 = axes[0]
ax2 = ax1.twinx()
ax3 = axes[1]

ax1.hist(v,density = True, color="r", bins= 100, label="Initial velocity distribution")
n,b,patches = ax2.hist(final_v.tolist(),weights=weight.tolist() ,density = True, bins = 100, label="Weighted velocity")
ax2.plot([v_sim, v_sim],[0, ax2.get_ylim()[1]], "k")
#ax2.plot(lnspc, pdf)
#for i,elem in enumerate(quantiles):
#    ax2.plot([elem, elem], [0, ax2.get_ylim()[1]], "-b")
#    ax2.text(elem, ax2.get_ylim()[0], r'$ %s %% $'%(int(percentiles[i]*100)))

ax1.legend(loc='upper right')
ax2.legend(loc='upper left')

ax3.errorbar(mjhd, lc_sim, yerr=err_mag_ml,ls ='',marker = 'o', label = "Data", zorder=0)
ax3.plot(mjhd, best_lc , "+", label ="Best fit found", zorder = 5)
ax3.text(54000, (max(best_lc) + min(best_lc)) / 2-0.1, r'$ \chi = %s $' % (min_chi))
ax3.text(54000, (max(best_lc) + min(best_lc)) / 2, r'$ v = %s $' % (best_v ))
#ax3.text(54000, (max(best_lc) + min(best_lc)) / 2-0.3, r'$ coords = (%s , %s), angle : %s $' % (final_x[arg], final_y[arg], final_theta[arg]))
ax3.set_xlabel('MJHD')
ax3.set_ylabel(r'$m_A - m_B$')
ax3.legend()
ax3.set_ylim(1.2*max(best_lc), 1.2*min(best_lc))
#.xlabel('MJHD')
#.ylabel('Mag')


#print "Most likely velocity : %s" % (b[np.where(n == n.max())][0])

plt.show()
#fig.savefig(resultdir + "fit_v%s_ntraj%s_othermap_5.png" % (v_sim, n_traj))
#fig.savefig(resultdir + "png/data/R2_wavy_n1_%s.png"%(n_traj) )

#with open(resultdir + 'results_R20_v%s_ntraj%s_othermap_4.pkl' % (v_sim, n_traj), 'wb') as handle:
with open(resultdir + 'pkl/data/R2_wavy_n1_%s.pkl'%(n_traj), 'wb') as handle:
    pkl.dump((final_v,chi, best_lc), handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump(( chi), handle, protocol=pkl.HIGHEST_PROTOCOL)
    pkl.dump((  best_lc), handle, protocol=pkl.HIGHEST_PROTOCOL)






sys.exit()

#==================================================================================================================

#sampling = np.arange(0, 8000, 500)
#x_start= np.repeat(sampling, len(sampling))
#y_start = np.tile(sampling, len(sampling))


LC = []

slopes = np.random.normal(0,len(final_map)/(10*lengthmin), nslopes)

print slopes
#slopes = [0.5]
start= timeit.default_timer()

for i,slope in enumerate(slopes):
    temp = draw_LC(10, slope, final_map, lengthmin)
    LC.extend(temp)


stop = timeit.default_timer()

print "+++++++++++++++++++++++++++++++++++++++++++++++++++"
print stop -start
print len(LC)
len_LC = len(LC)
#for i in range(len_LC):
#    LC.append(LC[i][::-1])

meanlen = []
#for i,elem in enumerate(LC):
#    meanlen.append(len(elem))

#print np.mean(meanlen)
#-------------------------------------------------
arrows = []
for i,trajectory in enumerate(database):
    temp = [trajectory[0], trajectory[1], trajectory[2], trajectory[3]]
    arrows.append(temp)


display_multiple_trajectory(arrows, map)
sys.exit()

arrow = [path_x[0],path_y[0], path_x[-1],path_y[-1]]
display_trajectory_bis(arrow,map)
#-------------------------------------------------
subsets = []
start2 = timeit.default_timer()
#for i, lc in enumerate(LC):
#    temp = compare(lc, lc_sim, err_mag_ml, mjhd, v_source_in_map)
#    subsets.append(temp)


n_cpu = multiprocessing.cpu_count()
print n_cpu
pool = multiprocessing.Pool(int(n_cpu))
parallel_compare = partial(compare,sim = LC, data = lc_sim, errdata = err_mag_ml, time = mjhd)


v_real = np.abs(np.random.normal(0,295,32))
print v_real
v_pxl = np.divide(np.multiply(100000*3600*24,v_real),cm_per_pxl)
#sys.exit()
res=pool.map(parallel_compare, v_pxl)
print "aaaaaaaaaaaaaaaaaaaaaaaa"
print len(res)
if len(res)>0:

    chi_min, final_lc, v_opt, result, all_v= find_min(res)

    stop2 = timeit.default_timer()

    print stop2-start
    plt.plot(mjhd, final_lc, "+")
    plt.text(57000, 0,r'$ \chi = %s $'%(chi_min))
    plt.text(57000, (max(final_lc)+min(final_lc))/2, r'$ v = %s $' % (v_opt*cm_per_pxl/(100000*3600*24)))
    plt.plot(mjhd, lc_sim,"ro")
    plt.xlabel('MJHD')
    plt.ylabel('Mag')
    plt.show()

    print result
    all_v = np.array(all_v)
    result = np.array(result)
    all_v = np.divide(np.multiply(all_v,cm_per_pxl),100000*3600*24)
    weight = np.exp(np.divide(np.negative(np.power(result,2)),2))
    plt.hist(all_v, weights=weight)
    #plt.hist(all_v,color='r')

    plt.show()

    os.chdir(resultdir)
    with open('results_R20_%sv.pkl'%(len(all_v)), 'wb') as handle:
        pkl.dump(all_v, handle, protocol=pkl.HIGHEST_PROTOCOL)
        pkl.dump(result, handle, protocol=pkl.HIGHEST_PROTOCOL)
