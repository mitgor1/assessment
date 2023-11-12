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

@njit
def line_energy(angles, current_energy, ix, nmax):
    #calculating the index of previous and next rows accounting for periodic boundary conditions
    ix_plus = (ix + 1) % nmax
    ix_minus = (ix - 1) % nmax

    #calculating the angle differences for the row
    anglediff_next = angles[ix, :] - angles[ix_plus, :]
    anglediff_prev = angles[ix, :] - angles[ix_minus, :]
    anglediff_left = angles[ix, :] - np.roll(angles[ix, :], -1)
    anglediff_right = angles[ix, :] - np.roll(angles[ix, :], 1)

    #taking the cosine of the angle differences
    cosdiff_next = np.cos(anglediff_next)
    cosdiff_prev = np.cos(anglediff_prev)
    cosdiff_left = np.cos(anglediff_left)
    cosdiff_right = np.cos(anglediff_right)

    #computing the energy contributions using the cosine of angle differences
    current_energy += 0.5 - 1.5 * cosdiff_next**2
    current_energy += 0.5 - 1.5 * cosdiff_prev**2
    current_energy += 0.5 - 1.5 * cosdiff_left**2
    current_energy += 0.5 - 1.5 * cosdiff_right**2

#=======================================================================

@jit(nopython=True)
def get_Qab(angles, nmax):
    # Initialize a 2x2 matrix Qab with zeros
    Qab = np.zeros((2, 2))
    
    # Create a 2x2 identity matrix
    delta = np.eye(2, 2)
    
    # Create a 2D array 'lab' by reshaping a stacked array of cosines and sines of 'angles'
    lab = np.vstack((np.cos(angles), np.sin(angles))).reshape(2, nmax, nmax)
    
    # Loop through matrix indices a and b (0 to 1)
    for a in range(2):
        for b in range(2):
            # Loop through indices i and j (0 to nmax-1)
            for i in range(nmax):
                for j in range(nmax):
                    # Update Qab[a, b] using the given formula
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    
    # Normalize Qab by dividing by (2 * nmax * nmax)
    Qab = Qab / (2 * nmax * nmax)
    
    # Return the resulting Qab matrix
    return Qab

#computing jit angles
@jit(nopython=True)
def get_order(order_array, nsteps):
    # Initialize an array 'order' of length 'nsteps + 1' with zeros
    order = np.zeros(nsteps + 1)
    
    # Loop through time steps 't' (0 to nsteps-1)
    for t in range(nsteps):
        # Compute eigenvalues and eigenvectors of the input 'order_array[t]'
        eigenvalues, eigenvectors = np.linalg.eig(order_array[t])
        
        # Store the maximum eigenvalue in 'order[t]'
        order[t] = eigenvalues.max()
    
    # Return the 'order' array
    return order

#=======================================================================

# Define a function with njit compilation enabled
@njit
def gen_noise_matrix(nmax):
    # Create an empty array 'noise_values' to store random noise values
    noise_values = np.empty(nmax**2, dtype=np.float64)
    
    # Loop through the indices of 'noise_values'
    for index in range(len(noise_values)):
        # Generate a random normal value and assign it to 'noise_values[index]'
        noise_values[index] = np.random.normal()
    
    # Reshape the 1D 'noise_values' array into a 2D 'noise_matrix'
    noise_matrix = noise_values.reshape((nmax, nmax))
    
    # Return the resulting 'noise_matrix'
    return noise_matrix


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



@njit
def MC_step(angles, Ts, nmax, energy_array, order_array, accept_ratio, it):
    rand_perturb = gen_noise_matrix(nmax)  # Generate random angles
    current_energy = np.zeros((2, nmax, nmax))

    # Perform the MC step sequentially for all rows
    MC_update(angles, rand_perturb, current_energy, Ts, nmax, accept_ratio, it)

    # Sum up the new current_energy to get the total energy
    energy_array[it] = np.sum(current_energy[1])
    # Calculate order parameter Q
    order_array[it] = get_Qab(angles, nmax)

    return angles, current_energy, energy_array, order_array, accept_ratio

#=======================================================================
def main(nsteps, nmax, temp, pflag):
    Ts = temp

    # Create arrays to store energy, acceptance ratio and averaged Q matrix
    energy_array = np.zeros(nsteps + 1)
    order_array = np.zeros((nsteps + 1, 2, 2))
    accept_ratio = np.zeros(nsteps + 1)

    # Initialize grid
    angles = initdat(nmax)
    current_energy = np.zeros((nmax, nmax))

    # Plot initial frame of lattice
    plotdat(angles, current_energy, pflag, nmax)

    initial_time = time.time()
    #getting time and doing MC_steps algorithm

    for it in range(nsteps):
        angles, current_energy, energy_array, order_array, accept_ratio = MC_step(angles, Ts, nmax, energy_array, order_array, accept_ratio, it)

    #calculating final time
    order = get_order(order_array, nsteps)
    final_time = time.time()
    runtime = final_time - initial_time


    #average_order = sum(order_array) / nsteps
    #print(average_order)

    # Final outputs
    print(f"{sys.argv[0]}: Size: {nmax}, Steps: {nsteps}, T*: {Ts:5.3f}, Order: {order[-1]:5.3f}, Time: {runtime:8.6f} s")
    # Plot final frame of lattice and generate output file
    plotdat(angles, current_energy[1], pflag, nmax)
    savedat(angles, nsteps, Ts, runtime, accept_ratio, energy_array, order, nmax)

    #interactive widgets making use of google scholar items
    #can move the sliders to whichever parameters and code will save the data for that specific run whilst also outputting their plots
interact(main, 
         nsteps=IntSlider(min=1, max=10000, step=1, value=50, description='MC Steps'),
         nmax=IntSlider(min=5, max=100, step=1, value=50, description='Lattice Size'),
         temp=FloatSlider(min=0.0, max=2.0, step=0.01, value=0.5, description='Reduced Temperature'),
         pflag=Dropdown(options=[('No Plot', 0), ('Plot Of Energy', 1), ('Plot of Angles', 2), ('Black Plot', 3)], value=2, description='Different Plots'))


