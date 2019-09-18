import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'normal',
        'size'   : 20}

matplotlib.rc('font', **font)


m = np.linspace(0.01, 100, 10000)
m2 = m[m>0.03]
m3 = m2[m2<3]
dm = np.diff(m,prepend=0)
mdm = np.power(m,-2.35)

mdm3 = 100*np.multiply(mdm, dm)
mdm4 = mdm3[m>0.03]
mdm4 = mdm4[m2<3]

mean_m = 0.3
miller = np.ones(len(m[m<=1]))
miller=np.append(miller,np.power(m[m>1],-2.35))

rescale = 0.158*(1/(np.log(10)*m[m==1]))*np.exp(-(np.log10(m[m==1])-np.log10(0.08))**2/(2*0.69**2))/mdm3[m==1]
chabrier = 0.158*(1/(rescale*np.log(10)*m[m<1]))*np.exp(-(np.log10(m[m<1])-np.log10(0.08))**2/(2*0.69**2))

chabrier = np.append(chabrier, np.power(m[m>=1],-2.3))

fig, ax = plt.subplots(1, 1, figsize=(15, 15))
ax.plot(m,mdm3, label = 'Salpeter IMF')
#ax.plot(m, miller , label = 'Miller-Scalo IMF')
#ax.plot(m, chabrier , label = 'Chabrier IMF')
ax.fill_between(m3, mdm4, 0*np.ones(len(m3)), alpha = 0.3, label='Range of stellar mass')
ax.plot(mean_m*np.ones(2),[0,mdm[m==mean_m]],'r',label ='Mean stellar mass $< M >$')
ax.set(xscale = 'log', yscale = 'log', xlabel = r'Mass [$M_{\odot}$]', ylabel = 'Mass Function')
plt.legend()
plt.show()
