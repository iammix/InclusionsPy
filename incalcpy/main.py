import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt

###################################
# Define Problem Parameters
###################################
# Pile Parameter
s = 3.1
r = 0.125
Ep = 30e6 * 0.35
gp = 24
Apile = np.pi * r ** 2
Wp = gp * Apile

# Soil Parameter
Es0 = 100e3
Es1 = 25e3 * 4
Es2 = 9e3 * 4
Es3 = 11e3 * 4

Asoil = (s ** 2) - Apile  # find the value
Ds = 2 * np.sqrt(Asoil / np.pi)
Dp = 2 * r

qs0 = 200.00
qs1 = 50.00
qs2 = 205.20
qs3 = 322.00

mt = 2
Em0 = 20.00e3
Em1 = 10.70e3
Em2 = 11.30e3
Em3 = 19.27e3

g1 = 18
g2 = 21
g3 = 21
##################################################################
# Test Set Values
"""
g1 = 18
g2 = 18
g3 = 18

qs1 = 125.20
qs2 = 125.20
qs3 = 125.20


Em1 = 19.27e3
Em2 = 19.27e3
Em2 = 19.27e3

Es1 = 11e3
Es2 = 11e3
Es3 = 11e3
"""

#####################################################################
Ws1 = g1 * Asoil
Ws2 = g2 * Asoil
Ws3 = g3 * Asoil
mq = 11
qp = 9000

# External Load
DF = 170 * s * s

# Solver Parameters
N = 10000
tol = 1e-3
iterMax = 100

from util.dataStructures import Properties

props = Properties()
props.TrussFirstS = Properties({'type': 'Truss', 'E': Es0, 'Area': Asoil})
props.TrussFirstP = Properties({'type': 'Truss', 'E': Es0, 'Area': Apile})
props.TrussE1 = Properties({'type': 'Truss', 'E': Es1, 'Area': Asoil})
props.TrussE2 = Properties({'type': 'Truss', 'E': Es2, 'Area': Asoil})
props.TrussE3 = Properties({'type': 'Truss', 'E': Es3, 'Area': Asoil})
props.TrussP = Properties({'type': 'Truss', 'E': Ep, 'Area': Apile})
props.SpringElem = Properties({'type': 'Spring', 'k': Es3})

#################
# Create NodeSets
#################

from fem.NodeSet import NodeSet

P_nodes = NodeSet()
S_nodes = NodeSet()
i = 0.00
z1 = 1
z2 = 400
while i <= 20.6:
    P_nodes.add(z1, [i, 0])
    S_nodes.add(z2, [i, 1])
    z1 += 1
    z2 += 1
    i += 0.10

#########################################
# Create Elements
#########################################

from fem.ElementSet import ElementSet

S_elements = ElementSet(S_nodes, props)
P_elements = ElementSet(P_nodes, props)

######################
# Create Pile Elements
######################

# z = 1
# i = 1
# while i <= 125:
#    P_elements.add(z, 'TrussP', [i, i + 1])
#    z += 1
#    i += 1

y = 0.0
z = 1
i = 1
while y <= 0.5:
    P_elements.add(z, 'TrussFirstP', [i, i + 1])
    y += 0.1
    z += 1
    i += 1
while y < 20.5:
    P_elements.add(z, 'TrussP', [i, i + 1])
    y += 0.1
    z += 1
    i += 1
######################
# Create Soil Elements
######################

y = 0.0
z = 400
i = 400
while y <= 0.5:
    S_elements.add(z, 'TrussFirstS', [i, i + 1])
    y += 0.1
    z += 1
    i += 1
while y <= 3.0:
    S_elements.add(z, 'TrussE1', [i, i + 1])
    y += 0.1
    z += 1
    i += 1
while y < 11.0:
    S_elements.add(z, 'TrussE2', [i, i + 1])
    y += 0.1
    z += 1
    i += 1
while y < 20.5:
    S_elements.add(z, 'TrussE3', [i, i + 1])
    y += 0.1
    z += 1
    i += 1

#################
# Create DOFSpace
#################

from fem.DofSpace import DofSpace

P_dofs = DofSpace(P_elements)
S_dofs = DofSpace(S_elements)

from util.dataStructures import GlobalData

P_globdat = GlobalData(P_nodes, P_elements, P_dofs)
S_globdat = GlobalData(S_nodes, S_elements, S_dofs)

################################
# Assembly Pile Stiffness Matrix
################################

from numpy import zeros, array
from fem.Assembly import assembleTangentStiffness

a = P_globdat.state
Da = P_globdat.Dstate
fint = zeros(len(P_dofs))

output = [[0., 0.]]

fint = zeros(len(P_dofs))

K_P, fint = assembleTangentStiffness(props, P_globdat)
K_S, fint = assembleTangentStiffness(props, S_globdat)

kp = K_P.toarray()
ks = K_S.toarray()

# delete Zero Rows and Columns for K_P matrix

for i in range(411, 0, -2):
    kp = np.delete(kp, i, axis=0)

for i in range(411, 0, -2):
    kp = np.delete(kp, i, axis=1)

# delete Zero Rows andColumns for K_S matrix
for i in range(411, 0, -2):
    ks = np.delete(ks, i, axis=0)
for i in range(411, 0, -2):
    ks = np.delete(ks, i, axis=1)
ks[205][205] += 1.0 * float(10000000.00) * ks[205][205]
kp[205][205] += 0.0 * float(100.00) * kp[205][205]

Z = np.zeros((206, 206), dtype=int)
K = np.asarray(np.bmat([[kp, Z], [Z, ks]]))
Kf = np.zeros((413, 413))

row_to_be_added = []
i = 0
while i <= 411:
    if i <= 205:
        if i == 0:
            row_to_be_added.append(1)
        else:
            row_to_be_added.append(0)
    else:
        if i == 206:
            row_to_be_added.append(-1)
        else:
            row_to_be_added.append(0)
    i += 1

i = 0
column_to_be_added = []
while i <= 412:
    if i <= 205:
        if i == 0:
            column_to_be_added.append(1)
        else:
            column_to_be_added.append(0)
    else:
        if i == 206:
            column_to_be_added.append(-1)
        else:
            column_to_be_added.append(0)
    i += 1

K_row = np.vstack((K, row_to_be_added))
K = np.hstack((K_row, np.atleast_2d(column_to_be_added).T))

dofs = []
for i in range(413):
    dofs.append(0.0)

dofs = np.zeros(413)
np.transpose(dofs)

iiter = 0.0
R = np.zeros(len(dofs))
# DF = 40 # External Load to the Pile

######################################
# Initialize date before NR iterations
######################################
Fext = []
lang = 0.0
dofs[412] = lang


def fext(i, qs, mt, Em, Ds, dofs):
    try:
        ex = np.exp(((-1.0 * (dofs[i] - dofs[206 + i])) * mt * Em) / (Ds * qs))
    except OverflowError:
        ex = float('inf')
    return -(1 - ex) * qs


def dfext(i, qs, mt, Em, Ds, dofs):
    try:
        ex = np.exp(((-1.0 * (dofs[i] - dofs[206 + i])) * mt * Em) / (Ds * qs))
    except OverflowError:
        ex = float('inf')
    return (ex) * (-1.0 * mt * Em) / (Ds)


Fp = np.zeros(206)
Fs = np.zeros(207)

#########################
# Initialize Force Vector
#########################
dofs = np.asarray(dofs)
i = 0
while i < 206:
    if i == 0:
        Fp[i] = 1.0 * DF + ((Wp + 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1 / 2)
        Kf[i][i] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
        Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    elif i <= 5:
        Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1
        Kf[i][i] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
        Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    elif i <= 30:
        Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs1, mt, Em1, Ds, dofs)) * 0.1
        Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs1, mt, Em1, Ds, dofs)) * 0.1
        Kf[i][i] = 2 * r * np.pi * dfext(i, qs1, mt, Em1, Ds, dofs) * 0.1
        Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs1, mt, Em1, Ds, dofs) * 0.1
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    elif i < 111:
        Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs2, mt, Em2, Ds, dofs)) * 0.1
        Kf[i][i] = 2 * r * np.pi * dfext(i, qs2, mt, Em2, Ds, dofs) * 0.1
        Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs2, mt, Em2, Ds, dofs) * 0.1
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    elif i < 205:
        Fp[i] = (Wp + r * r * np.pi * fext(i, qs3, mt, Em3, Ds, dofs)) * 0.1
        Kf[i][i] = 2 * r * np.pi * dfext(i, qs3, mt, Em3, Ds, dofs) * 0.1
        Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs3, mt, Em3, Ds, dofs) * 0.1
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    elif i == 205:
        Fp[i] = (Wp + r * r * np.pi * fext(i, qp, mq, Em3, Dp, dofs))
        Kf[i][i] = r * r * np.pi * dfext(i, qp, mq, Em3, Dp, dofs)
        Kf[i][i + 206] = -r * r * np.pi * dfext(i, qp, mq, Em3, Dp, dofs)
        Kf[i + 206][i] = Kf[i][i + 206]
        Kf[i + 206][i + 206] = Kf[i][i]
    i += 1

i = 0
while i < 206:
    if i == 0:
        Fs[i] = 0.0 * DF + (0.0 * Ws1 - 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2)
    elif i <= 5:
        Fs[i] = (0.0 * Ws1 - 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1
    elif i < 30:
        Fs[i] = (0.0 * Ws1 - 2 * r * np.pi * fext(i, qs1, mt, Em1, Ds, dofs)) * 0.1
    elif i < 111:
        Fs[i] = (0.0 * Ws2 - 2 * r * np.pi * fext(i, qs2, mt, Em2, Ds, dofs)) * 0.1
    elif i < 205:
        Fs[i] = (0.0 * Ws3 - 2 * r * np.pi * fext(i, qs3, mt, Em3, Ds, dofs)) * 0.1
    elif i == 205:
        Fs[i] = (0.0 * Ws3 - r * r * np.pi * fext(i, qp, mq, Em3, Dp, dofs))
    else:
        Fs[i] = 0  # lagrange force coef
    i += 1
Fext = np.concatenate((Fp, Fs), axis=0)

########################
# Incremental displacent
########################

du = np.zeros(len(dofs))
# dofs = np.asarray(dofs)


##################
# Start Iterations
##################

error = 1.0
tol = 1e-3
res = np.matmul(K, dofs) - Fext

while error > tol:
    du = np.zeros(len(dofs))
    du = np.linalg.solve(K - Kf, -res)
    check = np.linalg.cond(K, p='fro')
    dofs = dofs + du
    ##############################
    # Calculate new force vector
    ##############################
    i = 0
    while i <= 206:
        if i == 0:
            Fp[i] = 1.0 * DF + ((Wp + 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1 / 2)
            Kf[i][i] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
            Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
            Kf[i + 206][i] = Kf[i][i + 206]
        elif i <= 5:
            Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1
            Kf[i][i] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
            Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
            Kf[i + 206][i] = Kf[i][i + 206]
        elif i < 30:
            Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs1, mt, Em1, Ds, dofs)) * 0.1
            Kf[i][i] = 2 * r * np.pi * dfext(i, qs1, mt, Em1, Ds, dofs) * 0.1
            Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs1, mt, Em1, Ds, dofs) * 0.1
            Kf[i + 206][i] = Kf[i][i + 206]
        elif i < 111:
            Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs2, mt, Em2, Ds, dofs)) * 0.1
            Kf[i][i] = 2 * r * np.pi * dfext(i, qs2, mt, Em2, Ds, dofs) * 0.1
            Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs2, mt, Em2, Ds, dofs) * 0.1
            Kf[i + 206][i] = Kf[i][i + 206]
        elif i < 205:
            Fp[i] = (Wp + 2 * r * np.pi * fext(i, qs3, mt, Em3, Ds, dofs)) * 0.1
            Kf[i][i] = 2 * r * np.pi * dfext(i, qs3, mt, Em3, Ds, dofs) * 0.1
            Kf[i][i + 206] = -2 * r * np.pi * dfext(i, qs3, mt, Em3, Ds, dofs) * 0.1
            Kf[i + 206][i] = Kf[i][i + 206]
        elif i == 205:
            Fp[i] = (Wp + r * r * np.pi * fext(i, qp, mq, Em3, Dp, dofs))
            Kf[i][i] = r * r * np.pi * dfext(i, qp, mq, Em3, Dp, dofs)
            Kf[i][i + 206] = -r * r * np.pi * dfext(i, qp, mq, Em3, Dp, dofs)
            Kf[i + 206][i] = Kf[i][i + 206]

        i += 1

    i = 0
    while i <= 206:
        if i == 0:
            Fs[i] = 0.0 * DF + ((1.0 * Ws1 - 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1 / 2)
            Kf[i + 206][i + 206] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1 / 2
        elif i <= 5:
            Fs[i] = (0.0 * Ws1 - 2 * r * np.pi * fext(i, qs0, mt, Em0, Ds, dofs)) * 0.1
            Kf[i + 206][i + 206] = 2 * r * np.pi * dfext(i, qs0, mt, Em0, Ds, dofs) * 0.1
        elif i < 30:
            Fs[i] = (0.0 * Ws1 - 2 * r * np.pi * fext(i, qs1, mt, Em1, Ds, dofs)) * 0.1
            Kf[i + 206][i + 206] = 2 * r * np.pi * dfext(i, qs1, mt, Em1, Ds, dofs) * 0.1
        elif i < 111:
            Fs[i] = (0.0 * Ws2 - 2 * r * np.pi * fext(i, qs2, mt, Em2, Ds, dofs)) * 0.1
            Kf[i + 206][i + 206] = 2 * r * np.pi * dfext(i, qs2, mt, Em2, Ds, dofs) * 0.1
        elif i < 205:
            Fs[i] = (0.0 * Ws3 - 2 * r * np.pi * fext(i, qs3, mt, Em3, Ds, dofs)) * 0.1
            Kf[i + 206][i + 206] = 2 * r * np.pi * dfext(i, qs3, mt, Em3, Ds, dofs) * 0.1
        elif i == 205:
            Fs[i] = (0.0 * Ws3) - r * r * np.pi * fext(i, qp, mq, Em3, Dp, dofs)
            Kf[i + 206][i + 206] = r * r * np.pi * dfext(i, qp, mq, Em3, Dp, dofs)
            Fs[i] = (0.0 * Ws3) + qp * (dofs[i] - dofs[i + 206])
            # Kf[i+206][i+206]=r* r*np.pi*dfext(i, qp, mq, Em3, Dp, dofs)
        else:
            Fs[i] = 0  # lagrange force coef
        i += 1
    Fext = np.concatenate((Fp, Fs), axis=0)

    res = np.matmul(K, dofs) - Fext
    # res[205]-=float(1000.00)*dofs[205]
    error = np.linalg.norm(np.matmul(K, dofs) - Fext)

    # Increment NR counter

    iiter += 1

    print('Iter', iiter, ':', error)

    if iiter == iterMax:
        raise RuntimeError('Newton-Raphson iterations did not converge!')

###########################
# Calculate Internal Forces
###########################

P_axial = np.zeros(205)
z = 0
for i in range(204):
    if i <= 5:
        P_axial[i] = Es0 * Apile * (dofs[z + 1] - dofs[z]) / 0.1
    else:
        P_axial[i] = Ep * Apile * (dofs[z + 1] - dofs[z]) / 0.1

    z += 1

#############
# Plot curves
#############
x = []
y = []
x1 = []
for i in range(206):
    x.append(i / 10)
    x1.append(i / 10)
    if i <= 204:
        y.append(i / 10)

piledofs = []
soildofs = []
for i in range(206):
    piledofs.append(dofs[i])
    soildofs.append(dofs[206 + i])

###########
x1.append(20.9)
piledofs.append(0)
soildofs.append(0)
##########################################################
# Plots
###########################################################
# """
plt.figure()
plt.subplot(1, 3, 1)
plt.plot(piledofs, x1, linestyle='solid', label='Pile')
plt.plot(soildofs, x1, linestyle='dotted', label='Soil')
plt.title("Displacement")
plt.xlabel('Settlement (m)')
plt.ylabel('Depth(m)')
plt.ylim(21, 0)
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(Fext[:206], x, linestyle='solid', label='Pile')
plt.plot(Fext[206:412], x, linestyle='dotted', label='Soil')
plt.title(" External Forces ")
plt.xlabel('Force (kN)')
plt.ylabel('Depth(m)')
plt.ylim(21, 0)
plt.xlim(-50, 50)
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(P_axial, y, linestyle='solid', label='Pile')
plt.title('Axial Forces')
plt.xlabel('Forces(kN)')
plt.ylabel('Depth(m)')
plt.ylim(21, 0)
plt.legend()

plt.show()
# """
