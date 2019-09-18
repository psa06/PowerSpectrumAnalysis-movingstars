import matplotlib.pyplot as plt
import numpy as np
import itertools
import sys
from scipy import stats

v1 = []
v2 = []
v3 = []
n = 30



def resultante(v1,v2,v3):
    return v1+v2+v3



def projection_er(norme):
    proj = []
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, np.pi, n)
    comb = itertools.combinations_with_replacement(range(len(theta)), 2)
    for i,elem in enumerate(comb):
        proj.append(norme*np.sin(theta[elem[0]])*np.cos(phi[elem[1]]))

    return proj

def projection_ephi(norme):
    proj = []
    theta = np.linspace(0, 2*np.pi, n)
    phi = np.linspace(0, np.pi, n)
    comb = itertools.combinations_with_replacement(range(len(theta)), 2)
    for i,elem in enumerate(comb):
        proj.append(norme*np.sin(theta[elem[0]])*np.sin(phi[elem[1]]))

    return proj

def projection_etheta(norme):

    theta = np.linspace(0, 2*np.pi, n)
    return norme*np.cos(theta)


v1_r= projection_er(328)
v2_r= projection_er(277)
v3_r= projection_er(248)

v1_theta = projection_etheta(328)
v2_theta = projection_etheta(277)
v3_theta = projection_etheta(248)

v1_phi = projection_ephi(328)
v2_phi = projection_ephi(277)
v3_phi = projection_ephi(248)


comb_x = itertools.combinations_with_replacement(range(len(v1_r)),3)
comb_y = itertools.combinations_with_replacement(range(len(v1_phi)),3)
comb_z = itertools.combinations_with_replacement(range(len(v1_theta)),3)

result_x = []
result_y = []
result_z = []


for i,elem in enumerate(comb_x):
    result_x.append(resultante(v1_r[elem[0]],v2_r[elem[1]],v3_r[elem[2]]))


for i,elem in enumerate(comb_y):
    result_y.append(resultante(v1_phi[elem[0]],v2_phi[elem[1]],v3_phi[elem[2]]))

for i,elem in enumerate(comb_z):
    result_z.append(resultante(v1_theta[elem[0]],v2_theta[elem[1]],v3_theta[elem[2]]))

print result_z

plt.hist(result_x, bins=100, normed=1)

xt = plt.xticks()[0]
xmin, xmax = min(xt), max(xt)
lnspc = np.linspace(xmin, xmax, len(result_x))

m,s = stats.norm.fit(result_x)
pdf_g = stats.norm.pdf(lnspc, m, s)
plt.plot(lnspc, pdf_g, "b")
print m
print s

plt.hist(result_y, bins=100, normed=1)

xt_y = plt.xticks()[0]
xmin_y, xmax_y = min(xt_y), max(xt_y)
lnspc_y = np.linspace(xmin_y, xmax_y, len(result_y))

m_y,s_y = stats.norm.fit(result_y)
pdf_g_y = stats.norm.pdf(lnspc_y, m_y, s_y)
plt.plot(lnspc_y, pdf_g_y, "r")
print m_y
print s_y

#plt.hist(result_z, bins=100, normed=1)

plt.show()