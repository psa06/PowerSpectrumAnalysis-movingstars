import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
import astropy

storagedir = "/run/media/epaic/TOSHIBA EXT/maps/Q0158/"

font = {'family' : 'normal',
        'size'   : 15}



matplotlib.rc('font', **font)
execfile("useful_functions.py")

einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192
ld_per_pxl = cm_per_pxl/(30000000000*3600*24)
c = 3e5
h0 = 70
q0 = 0.5

def ztoD(z):
    return c/(h0*q0**2)*(z*q0+(q0-1)*(np.sqrt(2*q0*z+1)-1))/(1+z)**2

#dos = ztoD(1.29)
#dol = ztoD(0.317)
#dls = dos-dol

#print np.sqrt((328.7/1.317*dls/dol)**2+2*(203/1.317*dos/dol)**2)
#print np.sqrt((277/1.317*dos/dol)**2+(248/2.29)**2+(328.7/1.317*dls/dol)**2+2*(203/1.317*dos/dol)**2)
#print np.sqrt((277/1.317*dos/dol)**2+(248/2.29)**2)
#sys.exit()



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
        #database.append([path_x[0],path_y[0], path_x[-1], path_y[-1]])
        return lc, path_x, path_y


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

map = datadir+"Q0158/FML0.9/mapA-B_fml09_R20.fits"
img = astropy.io.fits.open(map)[0]
final_map = img.data[:, :]

i =0
while i <10000:
    #x0 = np.random.random_integers(2400, 3400)
    #y0 = np.random.random_integers(700, 1600)
    x0 =205
    y0=len(final_map)-1-7134
    i+=1
    #theta = np.random.uniform(0,2*np.pi)
    #theta = 0.222173554445
    #v_x_sim = np.multiply(v_sim,np.cos(theta))
    #v_x_sim = np.divide(np.multiply(100000*3600*24,v_x_sim),cm_per_pxl)
    v_x_sim = 0.03497750

    #v_y_sim = np.multiply(v_sim,np.sin(theta))
    #v_y_sim = np.divide(np.multiply(100000*3600*24,v_y_sim),cm_per_pxl)
    v_y_sim = 0.08381402

    if x0 + (mjhd[-1] - mjhd[0]) * v_x_sim <= len(map) and y0 + (mjhd[-1] - mjhd[0]) * v_y_sim <= len(map) and x0 + (mjhd[-1] - mjhd[0]) * v_x_sim >= 0 and y0 + (mjhd[-1] - mjhd[0]) * v_y_sim >= 0:
        break

v_sim = np.sqrt(v_x_sim**2+v_y_sim**2)*cm_per_pxl/(100000*3600*24)
lc_sim,path_x, path_y = sim_LC(x0,y0, v_x_sim, v_y_sim, mjhd,err_mag_ml, final_map)


n_traj = 200000
v = np.random.uniform(100,1000,n_traj)

final_v, chi, best_lc = pkl.load(open(resultdir + 'pkl/sim/R20_thin_disk_100000.pkl', 'rb'))
#final_v2, chi2, best_lc2 = pkl.load(open(resultdir + 'pkl/sim/R20_thin_disk_6000000_2.pkl', 'rb'))
#final_v3, chi3, best_lc3 = pkl.load(open(resultdir + 'pkl/data/fit_data_R20_fml09_3.pkl', 'rb'))


final_v = np.array(final_v)
#final_v=np.append(final_v,final_v2)
chi = np.array(chi)
#chi = np.append(chi,chi2)

print len(final_v)
print type(chi)
print type(best_lc)

weight = []

for i,elem in enumerate(chi):
    temp = np.exp(-elem/2)
    weight.append(temp)

weight = np.array(weight)
min_chi = min(chi)
arg = np.where(chi==min_chi)[0][0]
best_v = final_v[arg]

fig, axes = plt.subplots(1, 2, figsize=(20,9))
ax1 = axes[0]
ax2 = ax1.twinx()
ax3 = axes[1]

ax1.hist(v,density = True, color="r", bins= 100, label="Initial velocity distribution")
n,b,patches = ax2.hist(final_v.tolist(),weights=weight.tolist() ,density = True, bins = 100, label="Weighted velocity")
#ax2.plot([v_sim, v_sim],[0, ax2.get_ylim()[1]], "k")
#ax2.plot(lnspc, pdf)
#for i,elem in enumerate(quantiles):
#    ax2.plot([elem, elem], [0, ax2.get_ylim()[1]], "-b")
#    ax2.text(elem, ax2.get_ylim()[0], r'$ %s %% $'%(int(percentiles[i]*100)))

ax1.legend(loc='upper right')
ax2.legend(loc='upper left')
ymin = 0.1
ymax = 0.5
mjhdmax = 54500
ax3.errorbar([mjhd[0], mjhdmax], [ymin, ymax], yerr=[err_mag_ml[0], err_mag_ml[-1]],ls ='',marker = 'o', label = "Data", zorder=0)
ax3.plot([mjhd[0], mjhdmax], [ymin, ymax], "+", label ="Best fit found", zorder = 5)
#ax3.plot(mjhd, best_lc2 , "+", label ="Best fit found 2", zorder = 5)
#ax3.text(54000, (max(best_lc) + min(best_lc)) / 2-0.1, r'$ \chi = %s $' % (min_chi))
#ax3.text(54000, (max(best_lc) + min(best_lc)) / 2, r'$ v = %s $' % (best_v ))
#ax3.text(54000, (max(best_lc) + min(best_lc)) / 2-0.3, r'$ coords = (%s , %s), angle : %s $' % (final_x[arg], final_y[arg], final_theta[arg]))
ax1.set(xlim= (0, 1400))

ax3.set_xlabel('MJHD')
ax1.set_xlabel(r'$v_e$ [$km\cdot s^{-1}$]')
ax3.set_ylabel(r'$m_A - m_B$')
plt.setp(ax1.get_yticklabels(), visible=False)
plt.setp(ax2.get_yticklabels(), visible=False)
ax3.yaxis.tick_right()
ax3.yaxis.set_label_position('right')
ax3.legend()
#ax3.set_ylim(1.2*max(best_lc), 1.2*min(best_lc))
#.xlabel('MJHD')
#.ylabel('Mag')


#print "Most likely velocity : %s" % (b[np.where(n == n.max())][0])

plt.show()