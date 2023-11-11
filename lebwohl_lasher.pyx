# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False

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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
import sys
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cython
cimport numpy as np
cimport numpy as cnp
cimport openmp
from libc.math cimport exp
from libc.math cimport sin, cos, exp
from cython.parallel import prange
from cython cimport Py_ssize_t
from libc.math cimport exp, cos
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt, log, M_PI
from libc.math cimport cos, pow
from libc.time cimport time

def initdat(int nmax):
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
    cdef double[:, :] arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    return np.asarray(arr)

#=======================================================================
def plotdat(cnp.ndarray[cnp.float64_t, ndim=2] arr, int pflag, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  pflag (int) = parameter to control plotting;
      nmax (int) = side length of square lattice.
    Description:
      Function to make a pretty plot of the data array.  Makes use of the
      quiver plot style in matplotlib.  Use pflag to control style:
        pflag = 0 for no plot (for scripted operation);
        pflag = 1 for energy plot;
        pflag = 2 for angles plot;
        pflag = 3 for black plot.
	  The angles plot uses a cyclic color map representing the range from
	  0 to pi.  The energy plot is normalised to the energy range of the
	  current frame.
	Returns:
      NULL
    """
    if pflag==0:
        return

    cdef:
        int i, j
        cnp.ndarray[cnp.float64_t, ndim=2] u = np.cos(arr)
        cnp.ndarray[cnp.float64_t, ndim=2] v = np.sin(arr)
        cnp.ndarray[cnp.int_t, ndim=1] x = np.arange(nmax)
        cnp.ndarray[cnp.int_t, ndim=1] y = np.arange(nmax)
        cnp.ndarray[cnp.float64_t, ndim=2] cols = np.zeros((nmax, nmax))

    if pflag==1: # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        for i in range(nmax):
            for j in range(nmax):
                cols[i,j] = one_energy(arr,i,j,nmax)
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag==2: # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = arr%np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(arr)
        norm = plt.Normalize(vmin=0, vmax=1)

    quiveropts = dict(headlength=0,pivot='middle',headwidth=1,scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols,norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.savefig(f"plot_{nmax}.png")

#=======================================================================
def savedat(cnp.ndarray[cnp.float64_t, ndim=2] arr,int nsteps,double Ts,double runtime,cnp.ndarray[cnp.float64_t, ndim=1] ratio,cnp.ndarray[cnp.float64_t, ndim=1] energy,cnp.ndarray[cnp.float64_t, ndim=1] order,int nmax):          
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  nsteps (int) = number of Monte Carlo steps (MCS) performed;
	  Ts (float) = reduced temperature (range 0 to 2);
	  ratio (float(nsteps)) = array of acceptance ratios per MCS;
	  energy (float(nsteps)) = array of reduced energies per MCS;
	  order (float(nsteps)) = array of order parameters per MCS;
      nmax (int) = side length of square lattice to simulated.
    Description:
      Function to save the energy, order and acceptance ratio
      per Monte Carlo step to text file.  Also saves run data in the
      header.  Filenames are generated automatically based on
      date and time at beginning of execution.
	Returns:
	  NULL
    """
    # Create filename based on current date and time.
    current_datetime = datetime.datetime.now().strftime("%a-%d-%b-%Y-at-%I-%M-%S%p")
    filename = "LL-Output-{:s}-{}-{}.txt".format(current_datetime,nmax,Ts)
    FileOut = open(filename,"w")
    # Write a header with run parameters
    FileOut.write("#=====================================================\n")
    FileOut.write("# File created:        {:s}\n".format(current_datetime))
    FileOut.write("# Size of lattice:     {:d}x{:d}\n".format(nmax,nmax))
    FileOut.write("# Number of MC steps:  {:d}\n".format(nsteps))
    FileOut.write("# Reduced temperature: {:5.3f}\n".format(Ts))
    FileOut.write("# Run time (s):        {:8.6f}\n".format(runtime))
    FileOut.write("#=====================================================\n")
    FileOut.write("# MC step:  Ratio:     Energy:   Order:\n")
    FileOut.write("#=====================================================\n")
    # Write the columns of data
    for i in range(nsteps+1):
        FileOut.write("   {:05d}    {:6.4f} {:12.4f}  {:6.4f} \n".format(i,ratio[i],energy[i],order[i]))
    FileOut.close()
#=======================================================================

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    data = []
    for line in lines:
        if line.strip() and line[0].isdigit():  
            parts = line.split()
            
            step_data = {
                'MC step': int(parts[0]),
                'Ratio': float(parts[1]),
                'Energy': float(parts[2]),
                'Order': float(parts[3])
            }
            data.append(step_data)
            
    return data

def compare_values(reference_data, comparison_data, tolerance=0.05):
    match_counts = {'Energy': 0, 'Order': 0, 'Ratio': 0}
    entry_counts = {'Energy': 0, 'Order': 0, 'Ratio': 0}

    for ref_entry, comp_entry in zip(reference_data, comparison_data):
        
        if abs(ref_entry['Energy'] - comp_entry['Energy']) <= tolerance * abs(ref_entry['Energy']):
            match_counts['Energy'] += 1
        entry_counts['Energy'] += 1

        
        if abs(ref_entry['Order'] - comp_entry['Order']) <= tolerance * abs(ref_entry['Order']):
            match_counts['Order'] += 1
        entry_counts['Order'] += 1

        
        if abs(ref_entry['Ratio'] - comp_entry['Ratio']) <= tolerance * abs(ref_entry['Ratio']):
            match_counts['Ratio'] += 1
        entry_counts['Ratio'] += 1

    
    similarity_percentages = {
        category: (match_counts[category] / entry_counts[category]) * 100
        for category in match_counts
    }

    
    total_match_count = sum(match_counts.values())
    total_entry_count = sum(entry_counts.values())
    overall_similarity_percentage = (total_match_count / total_entry_count) * 100

    # Print the individual percentages and the overall percentage
    print(f"Closeness percentage for each value is:")
    for category, percent in similarity_percentages.items():
        print(f"{category}: {percent:.2f}%")
    print(f"Overall: {overall_similarity_percentage:.2f}%")
    
"""
file_1 = "LL-Output-{:s}-{}-{}.txt".format(current_datetime,nmax,Ts)
file_2 = 'comparison.txt'

data1 = read_file(file_1)
data2 = read_file(file_2)

comparison = compare_values(data1, data2)
"""

def get_Qab(cnp.ndarray[cnp.float64_t, ndim=2] angles, int nmax):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Qab = np.zeros((2, 2), dtype=np.float64)
    cdef double[:, :] delta = np.eye(2, dtype=np.float64)
    cdef double[:, :, :] lab = np.vstack((np.cos(angles), np.sin(angles))).reshape(2, nmax, nmax)
    cdef int a, b, i, j

    for a in range(2):
        for b in range(2):
            for i in range(nmax):
                for j in range(nmax):
                    Qab[a, b] += 3 * lab[a, i, j] * lab[b, i, j] - delta[a, b]
    Qab = Qab / (2 * nmax * nmax)
    return np.asarray(Qab)

#using a cdef function instead of direct manipulation for faster processing:
cdef cnp.ndarray[cnp.float64_t, ndim=2] gen_noise_matrixO(int nmax, double scale):
    # Declare variables with static types
    cdef int index
    cdef cnp.ndarray[cnp.float64_t, ndim=1] noise_values = np.empty(nmax**2, dtype=np.float64)

    # Loop for generating noise values
    for index in range(nmax**2):
        noise_values[index] = np.random.normal(scale=scale)

    # Reshape the noise array to a 2D matrix
    cdef cnp.ndarray[cnp.float64_t, ndim=2] noise_matrix = noise_values.reshape((nmax, nmax))

    return noise_matrix

#same calculations as the numba version and original
def get_order(cnp.ndarray[cnp.float64_t, ndim=3] order_array, int nsteps):
    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = np.zeros(nsteps + 1, dtype=np.float64)
    cdef int t
    for t in range(nsteps):
        eigenvalues,eigenvectors = np.linalg.eig(order_array[t])
        order[t] = eigenvalues.max()
    return np.asarray(order)

#=======================================================================

cdef void line_energy(cnp.ndarray[cnp.float64_t, ndim=2] angles, double[:, :, :] current_energy, int ix, int nmax, int old_new):
    # Assume angles is a 2D array of shape (nmax, nmax)
    cdef int iy, ix_plus, ix_minus
    cdef double anglediff_next, anglediff_prev, anglediff_left, anglediff_right
    cdef double cosdiff_next, cosdiff_prev, cosdiff_left, cosdiff_right

    # Calculate index of previous and next rows accounting for periodic boundary conditions
    ix_plus = (ix + 1) % nmax
    ix_minus = (ix - 1) % nmax

    for iy in range(nmax):
        # Calculate angle differences for the row using periodic boundary conditions for columns
        iy_plus = (iy + 1) % nmax
        iy_minus = (iy - 1) % nmax

        anglediff_next = angles[ix, iy] - angles[ix_plus, iy]
        anglediff_prev = angles[ix, iy] - angles[ix_minus, iy]
        anglediff_left = angles[ix, iy] - angles[ix, iy_minus]
        anglediff_right = angles[ix, iy] - angles[ix, iy_plus]

        # Take the cosine of the angle differences
        cosdiff_next = cos(anglediff_next)
        cosdiff_prev = cos(anglediff_prev)
        cosdiff_left = cos(anglediff_left)
        cosdiff_right = cos(anglediff_right)

        # Perform operations directly on the memory view slice
        current_energy[old_new, ix, iy] = (0.5 - 1.5 * pow(cosdiff_next, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_prev, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_left, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_right, 2))


cdef inline double random_const():
    return rand() / float(RAND_MAX)

cdef inline double energy_diff_calc(double[:, :] arr, int ix, int iy, int nmax, double ang):
    cdef:
        double placement = arr[ix, iy]
        double en_0 = one_energy(arr, ix, iy, nmax)
    arr[ix, iy] += ang
    cdef double en_1 = one_energy(arr, ix, iy, nmax)
    arr[ix, iy] = placement
    return en_1 - en_0

def MC_step(cnp.ndarray[cnp.float64_t, ndim=2] arr, double Ts, int nmax):
    """
    Arguments:
	  arr (float(nmax,nmax)) = array that contains lattice data;
	  Ts (float) = reduced temperature (range 0 to 2);
      nmax (int) = side length of square lattice.
    Description:
      Function to perform one MC step, which consists of an average
      of 1 attempted change per lattice site.  Working with reduced
      temperature Ts = kT/epsilon.  Function returns the acceptance
      ratio for information.  This is the fraction of attempted changes
      that are successful.  Generally aim to keep this around 0.5 for
      efficient simulation.
	Returns:
	  accept/(nmax**2) (float) = acceptance ratio for current MCS.
    """
    #
    # Pre-compute some random numbers.  This is faster than
    # using lots of individual calls.  "scale" sets the width
    # of the distribution for the angle changes - increases
    # with temperature.

    cdef:
        double scale = 0.1 + Ts
        int accept = 0
        int i, j, ix, iy
        double ang, en_diff, boltz
        cnp.ndarray[cnp.int_t, ndim=2] xran = np.random.randint(0, nmax, size=(nmax, nmax))
        cnp.ndarray[cnp.int_t, ndim=2] yran = np.random.randint(0, nmax, size=(nmax, nmax))
        cnp.ndarray[cnp.float64_t, ndim=2] aran = np.random.normal(scale=scale, size=(nmax, nmax))

    for i in range(nmax):
        for j in range(nmax):
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            en_diff = energy_diff_calc(arr, ix, iy, nmax, ang)
            if en_diff <= 0:
                arr[ix, iy] += ang
                accept += 1
            else:
                boltz = exp(-en_diff / Ts)
                if boltz >= random_const():
                    arr[ix, iy] += ang
                    accept += 1

    return accept / (nmax * nmax)

def main(int nsteps,int nmax,double temp,int pflag, int thread_count):
    """
    Arguments:
	  program (string) = the name of the program;
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
    cdef cnp.ndarray[cnp.float64_t, ndim=2] lattice = initdat(nmax)
    # Plot initial frame of lattice
    plotdat(lattice,pflag,nmax)
    # Create arrays to store energy, acceptance ratio and order parameter
    cdef cnp.ndarray[cnp.float64_t, ndim=1] energy = np.zeros(nsteps+1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] ratio = np.zeros(nsteps+1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = np.zeros(nsteps+1)
    
    # Set initial values in arrays
    energy[0] = all_energy(lattice,nmax,thread_count)
    ratio[0] = 0.5 # ideal value
    order[0] = get_order(lattice,nmax)

    # Begin doing and timing some MC steps.
    initial = openmp.omp_get_wtime()
    cdef int it
    for it in range(1,nsteps+1):
        ratio[it] = MC_step(lattice,temp,nmax)
        energy[it] = all_energy(lattice,nmax,thread_count)
        order[it] = get_order(lattice,nmax)
    final = openmp.omp_get_wtime()
    runtime = final-initial

    cdef double average_order = 0.0  
    average_order = sum(order) / nsteps
    #print(average_order)

    # Final outputs
    print("Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    savedat(lattice,nsteps,temp,runtime,ratio,energy,order,nmax)
    plotdat(lattice,pflag,nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 6:
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        THREADS = int(sys.argv[5])
        main(ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG, THREADS)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG> <THREADS>".format(sys.argv[0]))
#=======================================================================