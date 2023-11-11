"""
Basic Python Lebwohl-Lasher code.  Based on the paper 
P.A. Lebwohl and G. Lasher, Phys. Rev. A, 6, 426-429 (1972).
This version in 2D.

Run at the command line by typing:

python LebwohlLasher.py <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>

where:
  ITERATIONS = number of Monte Carlo steps, where 1MCS is when each cell
      has attempted a change once on average (i.e. SIZE*SIZE attempts)
  SIZE = side length of square lattice
  TEMPERATURE = reduced temperature in range 0.0 - 2.0.
  PLOTFLAG = 0 for no plot, 1 for energy plot and 2 for angle plot.
  
The initial configuration is set at random. The boundaries
are periodic throughout the simulation.  During the
time-stepping, an array containing two domains is used; these
domains alternate between old data and new data.

SH 16-Oct-23
"""

import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
from numba import jit, prange, cuda, njit
import numba 


#=======================================================================
@njit
def initdat(nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax,nmax).
    Description:
      Function to create and initialise the main data array that holds
      the lattice.  Will return a square lattice (size nmax x nmax)
	  initialised with random orientations in the range [0,2pi].
	Returns:
	  arr (float(nmax,nmax)) = array to hold lattice.
    """
    arr = np.random.random_sample((nmax,nmax))*2.0*np.pi
    return arr
#=======================================================================
def plotdat(angles,current_energy,pflag,nmax):
    if pflag==0:
        return
    u = np.cos(angles)
    v = np.sin(angles)
    x = np.arange(nmax)
    y = np.arange(nmax)
    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        #cols = np.asarray(current_energy[0, :])
        cols = current_energy
        #print("Shapes - u: ({}, {}), v: ({}, {}), x: {}, y: {}, cols: ({}, {})".format(u.shape[0], u.shape[1], v.shape[0], v.shape[1],x.size, y.size, cols.shape[0], cols.shape[1]))
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = angles%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(angles)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}.txt".format(current_datetime)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    print("#=====================================================",file=FileOut)
    print("# File created:        {:s}".format(current_datetime),file=FileOut)
    print("# Size of lattice:     {:d}x{:d}".format(nmax,nmax),file=FileOut)
    print("# Number of MC steps:  {:d}".format(nsteps),file=FileOut)
    print("# Reduced temperature: {:5.3f}".format(Ts),file=FileOut)
    print("# Run time (s):        {:8.6f}".format(runtime),file=FileOut)
    print("#=====================================================",file=FileOut)
    print("# MC step:  Ratio:     Energy:   Order:",file=FileOut)
    print("#=====================================================",file=FileOut)
    # Write the columns of data
    for i in range(nsteps+1):
        print("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} ".format(i,ratio[i],energy[i],order[i]),file=FileOut)
    FileOut.close()
#=======================================================================

@njit
def line_energy(angles, current_energy, ix, nmax):
    # Calculate index of previous and next rows accounting for periodic boundary conditions
    ix_plus = (ix + 1) % nmax
    ix_minus = (ix - 1) % nmax

    # Calculate angle differences for the row
    anglediff_next = angles[ix, :] - angles[ix_plus, :]
    anglediff_prev = angles[ix, :] - angles[ix_minus, :]
    anglediff_left = angles[ix, :] - np.roll(angles[ix, :], -1)
    anglediff_right = angles[ix, :] - np.roll(angles[ix, :], 1)

    # Now, take the cosine of the angle differences
    cosdiff_next = np.cos(anglediff_next)
    cosdiff_prev = np.cos(anglediff_prev)
    cosdiff_left = np.cos(anglediff_left)
    cosdiff_right = np.cos(anglediff_right)

    # Compute the energy contributions using the cosine of angle differences
    current_energy += 0.5 - 1.5 * cosdiff_next**2
    current_energy += 0.5 - 1.5 * cosdiff_prev**2
    current_energy += 0.5 - 1.5 * cosdiff_left**2
    current_energy += 0.5 - 1.5 * cosdiff_right**2

"""
@jit(nopython=True)
def one_energy(arr,ix,iy,nmax):
    en = 0.0
    ixp = (ix+1)%nmax # These are the coordinates
    ixm = (ix-1)%nmax # of the neighbours
    iyp = (iy+1)%nmax # with wraparound
    iym = (iy-1)%nmax #
#
# Add together the 4 neighbour contributions
# to the energy
#
    ang = arr[ix,iy]-arr[ixp,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ixm,iy]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iyp]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    ang = arr[ix,iy]-arr[ix,iym]
    en += 0.5*(1.0 - 3.0*np.cos(ang)**2)
    return en
"""
"""
@jit(nopython=True)
def one_energy(arr, ix, iy, nmax):
    en = 0.0
    constant = 0.5 * (1.0 - 3.0)
    
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    #attempted to optimise the performance of the energy calculation 
    #by taking the constant and repetitions out
    for dx, dy in [(ixp, iy), (ixm, iy), (ix, iyp), (ix, iym)]:
        ang = arr[ix, iy] - arr[dx, dy]
        en += constant * np.cos(ang) ** 2
    return en
"""

#=======================================================================
"""
@njit(parallel=True, fastmath=True)
def all_energy(arr, nmax):
    enall = 0.0
    for i in prange(nmax):
        for j in range(nmax):
            enall += one_energy(arr, i, j, nmax)
    return enall
"""
"""
def all_energy(arr,nmax):
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
"""    
#=======================================================================

@jit(nopython=True)
def get_Qab(angles,nmax):
    Qab = np.zeros((2,2))
    delta = np.eye(2,2)
    lab = np.vstack((np.cos(angles),np.sin(angles))).reshape(2,nmax,nmax)
    for a in range(2):
        for b in range(2):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    return Qab

@jit(nopython=True)
def get_order(order_array,nsteps):
    order = np.zeros(nsteps + 1)
    for t in range(nsteps):
        eigenvalues,eigenvectors = np.linalg.eig(order_array[t])
        order[t] = eigenvalues.max()
    return order

"""
@jit(nopython=True)
def get_order(arr,nmax):
    Qab = np.zeros((3,3))
    delta = np.eye(3,3)
    #
    # Generate a 3D unit vector for each cell (i,j) and
    # put it in a (3,i,j) array.
    #
    lab = np.vstack((np.cos(arr),np.sin(arr),np.zeros_like(arr))).reshape(3,nmax,nmax)
    for a in range(3):
        for b in range(3):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a,b] += 3*lab[a,i,j]*lab[b,i,j] - delta[a,b]
    Qab = Qab/(2*nmax*nmax)
    eigenvalues,eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()
"""
#=======================================================================

def precompute_randoms(nmax, Ts):
    scale = 0.1 + Ts
    xran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    yran = np.random.randint(0, high=nmax, size=(nmax, nmax))
    aran = np.random.normal(scale=scale, size=(nmax, nmax))
    pre_rand = np.random.uniform(0.0, 1.0, size=(nmax, nmax))
    return xran, yran, aran, pre_rand

@njit(parallel=True)
def MC_update(angles, rand_perturb, current_energy, Ts, nmax, accept_ratio, it):
    total_accept = 0 
    #line by line or row sequential updating 
    for ix in prange(nmax):
        #old energy computation for current row
        line_energy(angles, current_energy[0, ix, :], ix, nmax)

        #randome perturbation of row angles
        angles[ix, :] += rand_perturb[ix, :]

        #new row energy computation
        line_energy(angles, current_energy[1, ix, :], ix, nmax)

        #acceptance checks based off metrop factor criterias like before logic
        energy_diff = current_energy[1, ix, :] - current_energy[0, ix, :]
        metrop_factor = np.exp(-energy_diff / Ts)
        accept = (energy_diff <= 0) + ((energy_diff > 0) * (metrop_factor >= np.random.rand(nmax)))

        #appended based off acceptance
        current_energy[1, ix, :] = accept * current_energy[1, ix, :] + (1 - accept) * current_energy[0, ix, :]

        #angles changed based off acceptance as well and acceptence calculation done
        angles[ix, :] -= (1 - accept) * rand_perturb[ix, :]
        total_accept += np.sum(accept)

    #whole angles lattice acceptance:
    accept_ratio[it] += total_accept / (nmax**2)



@njit(parallel=True)
def MC_step(arr, Ts, nmax, xran, yran, aran, pre_rand):
    accept = np.zeros((nmax,), dtype=np.int32)

    for i in prange(nmax):
        for j in prange(nmax):
            ix, iy = xran[i, j], yran[i, j]
            ang = aran[i, j]
            en0 = one_energy(arr, ix, iy, nmax)
            arr[ix, iy] += ang
            en1 = one_energy(arr, ix, iy, nmax)

            energy_diff = en1 - en0
            if energy_diff <= 0 or np.exp(-energy_diff / Ts) >= pre_rand[i, j]:
                accept[i] += 1
            else:
                arr[ix, iy] -= ang

    return accept.sum() / (nmax * nmax)

#=======================================================================
def main(nsteps, nmax, temp, pflag):
    """
    Arguments:  
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square lattice to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    # Create and initialise lattice
    lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    energy = np.zeros(nsteps+1)
    ratio = np.zeros(nsteps+1)
    order = np.zeros(nsteps+1)
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = time.time()
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax)
        order[it] = get_order(lattice,nmax)
    final = time.time()
    runtime = final-initial
    
    
    print(f"The run was completed in {runtime:.2f} seconds.")
    
    savedat(lattice, nsteps, temp, runtime, ratio, energy, order, nmax)

    #interactive widgets making use of google scholar items
    #can move the sliders to whichever parameters and code will save the data for that specific run whilst also outputting their plots
interact(main, 
         nsteps=IntSlider(min=1, max=10000, step=1, value=50, description='MC Steps'),
         nmax=IntSlider(min=5, max=100, step=1, value=50, description='Lattice Size'),
         temp=FloatSlider(min=0.0, max=2.0, step=0.01, value=0.5, description='Reduced Temperature'),
         pflag=Dropdown(options=[('No Plot', 0), ('Plot Of Energy', 1), ('Plot of Angles', 2), ('Black Plot', 3)], value=2, description='Different Plots'))


