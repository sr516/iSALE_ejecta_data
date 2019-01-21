###############################################################################
######## Python script to calculate ejecta distribution #######################
###############################################################################
### Author: S. D. Raducan #####################################################
### Date: January 2019 ########################################################
###############################################################################

# Import libraries
import pySALEPlot as psp
import matplotlib.pyplot as plt
import numpy as np
import math as m
from scipy.stats import norm, binned_statistic
import numpy.polynomial.polynomial as poly
from scipy.optimize import leastsq, curve_fit
import scipy.integrate as integrate

# Create ouptut directory
dirname='Plots_output/'
psp.mkdir_p(dirname)

# Set up a pyplot figure
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,  figsize=(12,9))

# Open data file
# Replace 'name_file' as approptiate
moddir = ('name_file1',
          'name_file2',
          'name_file3',
          'name_file4')

label_name = ('Y$_{d0}$ = 100 kPa',
		      'Y$_{d0}$ = 10 kPa',
              'Y$_{d0}$ = 1 kPa',
	          'Y$_{d0}$ = 0.1 kPa')

colour = ('#f58f00', '#f35757', '#617BB7','#E06AAF')

# Open the data files.
for i in range(4):

	# Read simulation files
	# Replace name of the files as appropriate
    model=psp.opendatfile(moddir[i]+'/jdata.dat', tracermassvol=True)
    model.addRestartFile(moddir[i]+'regrid/jdata.dat')

    # Define simulation constants
    rho =  # Target density
    delta = # Impactor density
    U = # Impact velocity
    a = # Impactor radius
    m = 4./3.*np.pi*delta*a**3. # Impactor mass (if spherical)

    # Define numerical constants
    # Initial guess for constants
    c1 = # Approximation of C1 constant
    k = # Approximation of k constant
    mu = # Approximation of mu constant
    p = # Approximation of p constant
    q = # Approximation of q constant

    # Fixed constants
    nu = 0.4 # nu constant
    n1 = 1.2 # n1 constant
    n2 = 1.0 # n2 constant


    # Number of steps
    total_steps = model.totalsteps

    # Calculate final radius
    cg = model.craterGrowth(start=total_steps-1)
    radius = cg[-1,2]

    # Read the final step
    step = model.readStep(['TBx','TBy', 'TBr'],total_steps-1)
    vel_y = step.data[1]

    # Calculated ejection velocity (vel_ej) and launch position (x)
    vel_ej = np.sqrt(step.data[0]**2.+step.data[1]**2.)
    x = step.data[2]-2*a*step.data[0]/step.data[1]

    # Initialise arrays
    x_final, y_final = np.full((2,len(step.data[2])), 0.0), np.full((2,len(step.data[2])), 0.0)
    time = np.full(2, 0.0)

    l = 0
    # Read the final two steps
    for j in range(total_steps-2, total_steps, 1):
        step1 = model.readStep(['Trx', 'Try'], j)
        x_final[l, :] = step1.data[0]
        y_final[l, :] = step1.data[1]
        time[l] = step1.time
        l+=1

    # Calculate the final velocity of tracers
    v_x_final = abs(x_final[1,:]-x_final[0,:])/(time[1]-time[0])
    v_y_final = abs(y_final[1,:]-y_final[0,:])/(time[1]-time[0])
    vel_final = np.sqrt(v_x_final**2+v_y_final**2)

    # Exclude tracers outside the mesh
    for j in range(0, len(x_final[0,:]), 1):
        if y_final[0, j]>8:
            vel_final[j] = 100

    # Def function for mass
    mass_ej = model.tracerMass


    # Apply velocity threshold (v_thr)
    v_thr = 0.1

    # Apply threshold
    x, mass_ej, vel_y, vel_ej, vel_final =  x[vel_final>v_thr], mass_ej[vel_final>v_thr], vel_y[vel_final>v_thr], vel_ej[vel_final>v_thr], vel_final[vel_final>v_thr]

    # Sort and cumulative mass
    ind_mass = np.argsort(vel_ej)[::-1]
    mass_ej_sum = mass_ej[ind_mass].cumsum()

    # Plot m(v) vs v
    ax1.plot(vel_ej[ind_mass]/U, mass_ej_sum/m, colour[i], linewidth = 4., mec='None', alpha = 0.8, label=label_name[i])

    # Plot v vs x
    ax2.scatter(x[ind_mass]/a, vel_ej[ind_mass]/U, color = colour[i],  label = label_name[i])

    # Sort x and cumulative mass
    ind_x = np.argsort(x)
    mass_ej_sumx = mass_ej[ind_x].cumsum()

    # Plot m(v) vs x
    ax3.plot(x[ind_x]/a, mass_ej_sumx/m, colour[i], linewidth = 4., alpha = 0.8, mec='None', label=label_name[i])

    # Sort momentum
    ind_v = np.argsort(vel_y)[::-1]
    pt = mass_ej[ind_v]*vel_y[ind_v]
    ejp = pt.cumsum()
    beta = ejp[-1]/(m*U)+1

    # Plot momentum vs ejecta v_x
    ax4.plot(vel_y[ind_v]/U,ejp/(m*U), color = colour[i], label = label_name[i], linewidth = 4.)
    print ('Normalised ejected momentum from simulation, beta-1, is ', max(ejp)/(m*U))


    ###############################################################################
    # HH2011, HH2012, Cheng2016 fitted functions

    log_bins_for_x = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), num=200)

    # Binned statisctics
    bin1 = binned_statistic(x[ind], vel_ej[ind], 'median', bins=log_bins_for_x)
    vel_bin, x_bin = bin1[0], bin1[1]
    x_bin = x_bin[:-1]
    x_bin, vel_bin = x_bin[~np.isnan(vel_bin)], vel_bin[~np.isnan(vel_bin)]
    vel_bin, x_bin = vel_bin[~np.isnan(x_bin)], x_bin[~np.isnan(x_bin)] 

    # Define radius
    radius_med = (radius+x_bin.max())/2.

	# Define v/U as a function of x (Housen and Holsapple, 2011)
    def vdU_vs_x(xx, c1=c1, mu=mu):
        power = -1./mu
       	return np.log(c1*((xx/a)*(rho/delta)**nu)**power*abs(1.-xx/(n2*radius_med))**p*abs((n1*a)/xx-1.)**q) 

    # Find c1 and p from fitted function
    popt, pcov = curve_fit(vdU_vs_x, x_bin, np.log(vel_bin/U))

    c1, mu = popt[0], popt[1]

    print('Constants c1 and mu are: ', c1, np.sqrt(np.diag(pcov[0])[0,0]), mu, np.sqrt(np.diag(pcov[1])[1,1]))

    # Plot fitted function through binned data
    xr = np.linspace(n1*1.05*0.42, (n2*radius_med), 501)
    ax2.plot(xr/a, np.exp(vdU_vs_x(xr, c1, mu)), 'k--', linewidth = 3.)


    # Define function for mass ejected from within x (Housen and Holsapple, 2011)
    def Mdm_vs_xda(xda, k):
        return np.log(((3.*k)/(4.0*np.pi))*(rho/delta)*((xda/a)**3. - n1**3.))

    # Find k from fitted function
    popt, pcov = curve_fit(Mdm_vs_xda, x[ind_x], np.log(mass_ej_sumx/m) )
    k=popt[0]

    print('Constant k is: ', k, np.sqrt(np.diag(pcov[0])))

    # Plot fitted function
    ax3.plot(xr/a, np.exp(Mdm_vs_xda(xr, k)), 'k--', linewidth = 3.)

    # Plot ejecta mass as a function of velocity
    ax1.plot(np.exp(vdU_vs_x(xr, c1, mu)), np.exp(Mdm_vs_xda(xr, k)), 'k--', linewidth = 3.)

    # Calculate beta (Cheng et al., 2016)
    exp1 = (mu-nu)/mu
    exp2 = (3.*mu -1.)/mu
    beta_cal = 9.*k*c1/(4.*np.pi*2.**(1./2.))*(rho/delta)**exp1  * mu/(3.*mu-1.) * (  (0.74*n2*radius_med/a)**exp2 - (n1**exp2) )

    print ('Analythical beta-1 is: ',  beta_cal)

    # Plot analythical beta
    ax4.axhline((beta_cal), ls=':', c='r', linewidth = 2.)

    # Close files
    model.closeFile()

# Set axes labels
ax1.set_xlabel('Ejecta velocity, $v/U$', fontsize = 14)
ax1.set_ylabel('Cum. ejecta mass, $M(>v)/m$', fontsize = 14)

ax2.set_xlabel('Radial distance, $x/a$', fontsize = 14)
ax2.set_ylabel('Speed of ejecta, $v/U$', fontsize = 14)

ax3.set_xlabel('Radial distance, $x/a$', fontsize = 14)
ax3.set_ylabel('Cum. ejecta mass, $M(>x)/m$', fontsize = 14)

# Set axes labels
ax4.set_ylabel('Cum. ejecta momentum, $p_{ej(z)}(>v)/mU$', fontsize = 14)
ax4.set_xlabel('Ejecta velocity, $v_z/U$', fontsize = 14)

ax1.legend(loc='best', numpoints = 1, fontsize = 14)
ax2.legend(loc=3, numpoints = 1, fontsize = 14, scatterpoints = 1)
ax3.legend(loc='best', numpoints = 1, fontsize = 14)
ax4.legend(loc='best', numpoints = 1, fontsize = 14)

#ax1.annotate(r'$\frac{M}{m} = \frac{3k}{4\pi}C_1^{3\mu}[\frac{v}{U}(\frac{\rho}{\delta})^{\frac{3\nu-1}{3\mu}}]^{-3\mu}$',xy=(2.e-5,2.e-2),fontsize = 16.5, rotation=0)

#ax2.annotate(r'$\frac{v(x)}{U}=C_1[\frac{x}{a}(\frac{\rho}{\delta})^\nu]^{-\frac{1}{\mu}}(1-\frac{x}{n_2R})^p$',xy=(1.1,0.35),fontsize = 16.5, rotation=0)

#ax3.annotate(r'$\frac{M}{m} = \frac{3k}{4\pi}\frac{\rho}{\delta}[(\frac{x}{a})^3-n_1^3]$',xy=(7.,1.5e-2),fontsize = 16.5, rotation=0)

#Make plots logarithmic
ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xscale('log')
ax2.set_yscale('log')

ax3.set_xscale('log')
ax3.set_yscale('log')

ax4.set_xscale('log')


ax1.set_xlim([0.00001,1])

ax2.set_xlim([0.9,100])
ax2.set_ylim([0.00001,1.2])


fig.tight_layout()
#Save figure
plt.savefig('./Plots_output/integration.png', bbox_inches='tight')

#Close plot
plt.close(fig)
