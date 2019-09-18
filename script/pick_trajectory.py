from __future__ import division
from Tkinter import *
from PIL import ImageTk
from PIL import Image
import os
#from variousfct import *
import numpy as np
import matplotlib
import f2n
matplotlib.use("TkAgg")
from astropy.io import fits
from scipy.signal import lombscargle, periodogram
import pycs


einstein_r_03 = 3.41e16
cm_per_pxl = 20*einstein_r_03/8192

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

font = {'family' : 'normal',
        'size'   : 20}
matplotlib.rc('xtick', labelsize = 20)
matplotlib.rc('ytick', labelsize = 20)

matplotlib.rc('font', **font)

execfile("useful_functions.py")
mapdir = "Q0158/FML0.9M0.3/"
storagedir = "/run/media/epaic/TOSHIBA EXT/maps/Q0158/"

fitsfile = os.path.join(storagedir,"FML0.9/M0,3/mapA-B_fml09_R20.fits")  # path to the img_skysub.fits you will display
image = os.path.join(datadir, "mapA-B_fml09_R20.png")  # path to the png that will be created from the img_skysub.fits

#z1 = -1
#z2 = 7
#f2nimg = f2n.fromfits(fitsfile)
#f2nimg.setzscale(z1='ex',z2='ex')
#f2nimg.makepilimage(scale = "log", negative = False)
#f2nimg.tonet(image)

compteurs = 0  # first term counts the number of stars selected the second one the number of clicks you did to select the lens region and the third the same number for the empty region

trajcoord = []  # at the end it will contain the coordinates of the botom left corner and the top right corner of the lense region in the following order [x_bl, y_bl, x_tr, y_tr]


# The dimensions of the png
im = Image.open(image)
width, height = im.size

einstein_r = 3.414e16 #cm
cm_per_pxl = (20*einstein_r)/8192



def sim_LC(x_start, y_start, v_x, v_y, mjhd, err_data, map):
    #    x_start,y_start : Lists of integers. Lists of a
    # l the starting coordinates you want to test
    #   v_x, v_y : Lists of integers. Lists of all the coordinate of the velocity vector of the source on the lense plane you want to test (for now in pxl/point)
    #   n_points : Integer. Length of LC you want to simulate.
    #   map : Array of the convoluted map

    # One element of x_start, y_start, v_x and v_y is used to create a single lightcurve. If those parameters are lists instead of integers you will get list of lightcurves stored in lc. mjhd is a single list valid for every lightcurve.

    if x_start + (mjhd[-1] - mjhd[0]) * v_x <= len(map) and y_start + (mjhd[-1] - mjhd[0]) * v_y <= len(map) and x_start + (mjhd[-1] - mjhd[0]) * v_x >= 0 and y_start + (mjhd[-1] - mjhd[0]) * v_y >= 0:
        if v_x == 0:
            path_x = x_start * np.ones(len(mjhd))
        else:
            path_x = np.add(np.multiply(np.add(mjhd, -mjhd[0]),v_x), x_start)
        if v_y == 0:
            path_y = y_start * np.ones(len(mjhd))
        else:
            path_y = np.add(np.multiply(np.add(mjhd, -mjhd[0]), v_y), y_start)

        path_x= path_x.astype(int)
        path_y = path_y.astype(int)
        temp = np.multiply(-2.5, np.log10(map[path_y, path_x]))  # -2.5 log() to convert flux into mag


        lc = temp - temp[0] * np.ones(len(temp))

        return lc

def good_draw_LC(params, map, time,cm_per_pxl):
    x_start = params[0]
    y_start = params[1]
    #v = params[2]
    #angle= params[3]
    v_x = params[2]
    v_y = params[3]
    #v_x = np.multiply(v, np.cos(angle))
    #v_x = np.divide(np.multiply(100000 * 3600 * 24, v_x), cm_per_pxl)

    #v_y = np.multiply(v, np.sin(angle))
    #v_y = np.divide(np.multiply(100000 * 3600 * 24, v_y), cm_per_pxl)


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

    if path_x[-1] <= len(map)-1 and path_y[-1] <= len(map)-1 and path_x[-1] >= 0 and path_y[-1] >= 0:
        #temp = np.add(np.multiply(-2.5, np.log10(map[path_y, path_x])),
        #              np.random.normal(0, np.mean(err_data), len(path_y)))# -2.5 log() to convert flux into mag
        temp = np.multiply(-2.5, np.log10(map[path_y, path_x]))
        lc = temp - temp[0] * np.ones(len(temp))
        return lc



f = open(datadir+"microlensing/data/J0158_Euler_microlensing_upsampled_B-A.rdb","r")
f= f.read()
f=f.split("\n")
data = f[2:]

mjhd = np.array([])
err_mag_ml = np.array([])
mag_ml = np.array([])
window = 'flattop'
fcut = 0.01
sampling =1

img = fits.open(fitsfile)[0]
final_map = img.data[:, :]


for i,elem in enumerate(data):
    mjhd = np.append(mjhd,float(elem.split("\t")[0]))
    mag_ml = np.append(mag_ml, float(elem.split("\t")[1]))
    temp = elem.split("\t")[2]
    err_mag_ml= np.append(err_mag_ml,float(temp.split("\r")[0]))



new_mjhd = np.arange(mjhd[0], mjhd[-1], sampling)
lc = pycs.gen.lc.factory(mjhd, mag_ml, magerrs=err_mag_ml)
spline = pycs.gen.spl.fit([lc], knotstep=70, bokeps=20, verbose=False)
new_magml = spline.eval(new_mjhd)

frequency_spline, power_spline = periodogram(new_magml+  np.random.normal(0, np.mean(err_mag_ml), len(new_mjhd)), 1/sampling,window=window)
frequency_spline = np.array(frequency_spline[1:])
power_spline = np.array(power_spline[1:])


#frequency_spline = np.linspace(1/(2*(mjhd[-1]-mjhd[0])), 1/50, 50)

class LoadImage(Frame):
    def __init__(self, root):
        global mjhd
        global err_mag_ml
        global frame
        global ax
        global f
        global final_map
        global cm_per_pxl
        global window
        global fcut
        global sampling

        Frame.__init__(self, root)

        frame = Frame(root)
        # Creation of the canvas
        self.canvas = Canvas(frame, width=1000, height=1000, relief=SUNKEN)

        frame.pack()

        # display of the image
        File = image
        self.orig_img = Image.open(File)
        self.img = ImageTk.PhotoImage(self.orig_img)
        self.canvas.create_image(0, 0, image=self.img, anchor="nw")

        self.zoomcycle = 0
        self.zimg_id = None


        # Creation of scrollbars and shortcuts
        sbarV = Scrollbar(frame, orient=VERTICAL)
        sbarH = Scrollbar(frame, orient=HORIZONTAL)

        sbarV.config(command=self.canvas.yview)
        sbarH.config(command=self.canvas.xview)
        self.canvas.config(yscrollcommand=sbarV.set)
        self.canvas.config(xscrollcommand=sbarH.set)

        sbarV.pack(side=LEFT, fill=Y)
        sbarH.pack(side=BOTTOM, fill=X)
        root.bind("<Button-3>", self.select)
        root.bind("<Button-4>", self.zoomer)
        root.bind("<Button-5>", self.zoomer)
        self.canvas.bind("<Motion>", self.crop)
        self.canvas.pack(side=RIGHT, expand=True, fill=BOTH)
        self.canvas.config(scrollregion=(0, 0, width, height))
        self.canvas.config(highlightthickness=0)

    # if you are on windows, Button-4 and Button-5 are united under MouseWheel and instead of event.num = 5 or 4 you have event.delta = -120 ou 120
    def zoomer(self, event):
        if (event.num == 4):
            if self.zoomcycle != 5: self.zoomcycle += 1
        elif (event.num == 5):
            if self.zoomcycle != 0: self.zoomcycle -= 1
        self.crop(event)

    def crop(self, event):
        if self.zimg_id: self.canvas.delete(self.zimg_id)
        tmp = None
        if (self.zoomcycle) != 0:
            x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(
                event.y)  # to get the real coordinate of the star even after scrolling
            if self.zoomcycle == 1:
                tmp = self.orig_img.crop((x - 45, y - 30, x + 45, y + 30))
            elif self.zoomcycle == 2:
                tmp = self.orig_img.crop((x - 30, y - 20, x + 30, y + 20))
            elif self.zoomcycle == 3:
                tmp = self.orig_img.crop((x - 32, y - 32, x + 32, y + 32))
            elif self.zoomcycle == 4:
                tmp = self.orig_img.crop((x - 15, y - 10, x + 15, y + 10))
            elif self.zoomcycle == 5:
                tmp = self.orig_img.crop((x - 6, y - 4, x + 6, y + 4))
        if self.zoomcycle == 3:
            size = 300, 300
        else:
            size = 300, 200
            if tmp is not None:
                self.zimg = ImageTk.PhotoImage(tmp.resize(size))
                self.zimg_id = self.canvas.create_image(x, y, image=self.zimg)

    def select(self, event):
        global compteurs
        global choice
        global fact
        global height
        global trajcoord
        global bl_corner
        global bl_text
        global carre
        global final_map
        global window
        global fcut
        global sampling
        global cm_per_pxl

        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(
            event.y)  # to get the real coordinate of the star even after scrolling

        print "selection of the trajectory"
        print x
        print y
        compteurs += 1
        if compteurs % 3 == 1:              
            bl_corner = self.canvas.create_oval(x + 5, y - 5, x - 5, y + 5, outline="red")
            bl_text = self.canvas.create_text(x + 6, y + 6, text="Start of the trajectory", fill="green")
            trajcoord.append(x)
            trajcoord.append(y)
        elif compteurs % 3 == 2:
            self.canvas.delete(bl_corner)
            self.canvas.delete(bl_text)
            trajcoord.append(x)
            trajcoord.append(y)

            carre = self.canvas.create_line(trajcoord[0], trajcoord[1], trajcoord[2], trajcoord[3], width = 5, fill = 'red',
                                                 arrow = LAST)
            vx = (trajcoord[2]-trajcoord[0])/(mjhd[-1] - mjhd[0])
            vy = (trajcoord[1] - trajcoord[3])/(mjhd[-1] - mjhd[0])
            print trajcoord
            print vx
            print vy
            lc_sim = sim_LC(trajcoord[0], len(final_map)-1-trajcoord[1], vx, vy,mjhd, err_mag_ml, final_map)
            new_lc_sim = good_draw_LC([trajcoord[0], len(final_map) - 1 - trajcoord[1], vx, vy], final_map,new_mjhd,cm_per_pxl)
            print new_lc_sim
            power,frequency= periodogram(new_lc_sim, 1/sampling, window = 'flattop', detrend = 'constant')
            power= np.array(power)
            frequency= np.array(frequency)
            power = power[frequency >1/(len(new_mjhd))]
            frequency = frequency[frequency > 1 / (len(new_mjhd))]
            print power
            print frequency
            t = Toplevel(self)
            t.wm_title("Window 1")




            f,ax = plt.subplots(1, 1, figsize=(5, 5))
            #a = f.add_subplot(111)
            ax.plot(new_mjhd, new_lc_sim, "o")
            ax.text(55000, max(lc_sim)-0.1, r' v = %s $km\cdot s^{-1}$' % (int(np.sqrt(vx**2+vy**2)*cm_per_pxl/(100000*3600*24))))
            ax.set(ylim=[max(lc_sim), min(lc_sim)], xlabel= "MJHD", ylabel = "Magnitude")

            #ax[1].plot(frequency[frequency<fcut], power[frequency<fcut])
            #ax[1].plot(frequency_spline, power_spline, label='Data')
            #ax[1].set(xlabel=r"Frequency [days$^{-1}$]", ylabel="Power", yscale = "log", xscale = 'log')
            #locs = ax[1].get_xticks()
            #locs[1] = frequency_spline[0]

#            temp_lab = ax[1].get_xticklabels()
 #           lab = np.divide(1, locs).astype(int)
  #          labels = []
   #         for label in lab[1:-1]:
    #            labels.append('$\\frac{{{:2.0f}}}{{{:2.0f}}}$'.format(1, int(label)))

                # labels[0]='0'
     #       print locs
      #      ax[1].set_xticks(locs[1:-2], minor=False)
       #     ax[1].set_xticklabels(labels, minor=False)


            canvas = FigureCanvasTkAgg(f, master=t)
            canvas.show()
            canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)



        elif compteurs % 3 == 0:
            self.canvas.delete(carre)
            trajcoord = []





if __name__ == '__main__':
    root1 = Tk()
    root = LoadImage(root1)

    root.mainloop()