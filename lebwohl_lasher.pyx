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

# Define a Cython function to initialize a lattice
def initdat(int nmax):
    """
    Arguments:
      nmax (int) = size of lattice to create (nmax, nmax).
    Description:
      Function to create and initialize the main data array that holds
      the lattice. Will return a square lattice (size nmax x nmax)
      initialized with random orientations in the range [0, 2π].
    Returns:
      arr (float[nmax, nmax]) = array to hold lattice.
    """
    # Declare a memory view for a double precision 2D array 'arr'
    # This memory view efficiently accesses the underlying array
    cdef double[:, :] arr = np.random.random_sample((nmax, nmax)) * 2.0 * np.pi
    
    # Convert the Cython memory view to a NumPy array and return it
    return np.asarray(arr)

def plotdat(cnp.ndarray[cnp.float64_t, ndim=2] angles,double[:, :, :] current_energy,int pflag,int nmax):
    
    # Only proceed if pflag is not zero
    if pflag == 0:
        return
    
    # Perform the calculations
    cdef cnp.ndarray[cnp.float64_t, ndim=2] u = np.cos(angles)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] v = np.sin(angles)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] x = np.arange(nmax, dtype=np.int64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] y = np.arange(nmax, dtype=np.int64)


    # Handle the pflag logic
    if pflag == 1:  # colour the arrows according to energy
        mpl.rc('image', cmap='rainbow')
        #cols = np.asarray(current_energy).flatten()
        # Now cols will have the same total number of elements as u and v
        #assert cols.shape[0] == u.size  # This should be true if current_energy was initially the same shape as u and v
        #print("Shapes - u: ({}, {}), v: ({}, {}), x: {}, y: {}, cols: ({}, {})".format(u.shape[0], u.shape[1], v.shape[0], v.shape[1],x.size, y.size, cols.shape[0], cols.shape[1]))
        cols = np.asarray(current_energy[0, :, :])  # This slices out the first 2D layer of the 3D array.
        #cols = np.asarray(current_energy[1, :, :])  # This slices out the second 2D layer of the 3D array.
        #cols = np.asarray(current_energy)
        # Convert to a NumPy array to use `min` and `max`
        norm = plt.Normalize(cols.min(), cols.max())
    elif pflag == 2:  # colour the arrows according to angle
        mpl.rc('image', cmap='hsv')
        cols = angles % np.pi
        norm = plt.Normalize(vmin=0, vmax=np.pi)
    else:
        mpl.rc('image', cmap='gist_gray')
        cols = np.zeros_like(angles)
        norm = plt.Normalize(vmin=0, vmax=1)
    
    # Create the quiver plot
    quiveropts = dict(headlength=0, pivot='middle', headwidth=1, scale=1.1*nmax)
    fig, ax = plt.subplots()
    q = ax.quiver(x, y, u, v, cols, norm=norm, **quiveropts)
    ax.set_aspect('equal')
    plt.show()

#=======================================================================
def savedat(cnp.ndarray[cnp.float64_t, ndim=2] arr,int nsteps,double Ts,double runtime,cnp.ndarray[cnp.float64_t, ndim=1] ratio,cnp.ndarray[cnp.float64_t, ndim=1] energy,cnp.ndarray[cnp.float64_t, ndim=1] order,int nmax):        
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

# Define a function to read data from a file and parse it into a list of dictionaries
def read_file(file_path):
    # Open the file for reading
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Initialize an empty list to store the parsed data
    data = []
    
    # Loop through each line in the file
    for line in lines:
        # Check if the line is not empty and starts with a digit
        if line.strip() and line[0].isdigit():  
            # Split the line into parts based on whitespace
            parts = line.split()
            
            # Create a dictionary to store the parsed data for each step
            step_data = {
                'MC step': int(parts[0]),
                'Ratio': float(parts[1]),
                'Energy': float(parts[2]),
                'Order': float(parts[3])
            }
            
            # Append the step data to the list
            data.append(step_data)
    
    # Return the parsed data as a list of dictionaries
    return data

# Define a function to compare values between two datasets with a specified tolerance
def compare_values(reference_data, comparison_data, tolerance=0.05):
    # Initialize dictionaries to store counts of matches and entries for each category
    match_counts = {'Energy': 0, 'Order': 0, 'Ratio': 0}
    entry_counts = {'Energy': 0, 'Order': 0, 'Ratio': 0}

    # Iterate through corresponding entries in the reference and comparison datasets
    for ref_entry, comp_entry in zip(reference_data, comparison_data):
        
        # Check if the energy difference is within the specified tolerance
        if abs(ref_entry['Energy'] - comp_entry['Energy']) <= tolerance * abs(ref_entry['Energy']):
            match_counts['Energy'] += 1
        entry_counts['Energy'] += 1

        # Check if the order difference is within the specified tolerance
        if abs(ref_entry['Order'] - comp_entry['Order']) <= tolerance * abs(ref_entry['Order']):
            match_counts['Order'] += 1
        entry_counts['Order'] += 1

        # Check if the ratio difference is within the specified tolerance
        if abs(ref_entry['Ratio'] - comp_entry['Ratio']) <= tolerance * abs(ref_entry['Ratio']):
            match_counts['Ratio'] += 1
        entry_counts['Ratio'] += 1

    # Calculate similarity percentages for each category
    similarity_percentages = {
        category: (match_counts[category] / entry_counts[category]) * 100
        for category in match_counts
    }

    # Calculate the overall similarity percentage
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

#broken the original get_order into two to work with the mc_update and steps algorithms
# Define a Cython function to calculate Qab matrix
def get_Qab(cnp.ndarray[cnp.float64_t, ndim=2] angles, int nmax):
    # Initialize a 2x2 matrix Qab with zeros
    cdef cnp.ndarray[cnp.float64_t, ndim=2] Qab = np.zeros((2, 2), dtype=np.float64)
    
    # Create a 2x2 identity matrix
    cdef double[:, :] delta = np.eye(2, dtype=np.float64)
    
    # Create a 2D array 'lab' by reshaping a stacked array of cosines and sines of 'angles'
    cdef double[:, :, :] lab = np.vstack((np.cos(angles), np.sin(angles))).reshape(2, nmax, nmax)
    
    cdef int a, b, i, j
    
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
    
    # Convert the Cython array to a NumPy array and return it
    return np.asarray(Qab)


"""
AFTER TESTING: it makes code even slower... due to multiple numpy processes no point.
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
"""

# Define a Cython function to compute the order parameter
def get_order(cnp.ndarray[cnp.float64_t, ndim=3] order_array, int nsteps):
    #Initialize array order of length for nsteps to iterate
    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = np.zeros(nsteps + 1, dtype=np.float64)
    cdef int t
    
    #Looping through time steps 't'
    for t in range(nsteps):
        # Computing eigenvalues and eigenvectors of the input 
        eigenvalues, eigenvectors = np.linalg.eig(order_array[t])
        
        # Store the maximum eigenvalue in 'order[t]'
        order[t] = eigenvalues.max()
    
    #convert the cython array to a numPy array and return it
    return np.asarray(order)

#=======================================================================

cdef void line_energy(cnp.ndarray[cnp.float64_t, ndim=2] angles, double[:, :, :] current_energy, int ix, int nmax, int old_new):
    #assume angles is a 2D array of shape (nmax, nmax)
    cdef int iy, ix_plus, ix_minus
    cdef double anglediff_next, anglediff_prev, anglediff_left, anglediff_right
    cdef double cosdiff_next, cosdiff_prev, cosdiff_left, cosdiff_right

    #calculate index of previous and next rows accounting for the periodic boundary conditions
    ix_plus = (ix + 1) % nmax
    ix_minus = (ix - 1) % nmax

    for iy in range(nmax):
        #calculate the angle differences for the row using periodic boundary conditions for columns
        iy_plus = (iy + 1) % nmax
        iy_minus = (iy - 1) % nmax

        anglediff_next = angles[ix, iy] - angles[ix_plus, iy]
        anglediff_prev = angles[ix, iy] - angles[ix_minus, iy]
        anglediff_left = angles[ix, iy] - angles[ix, iy_minus]
        anglediff_right = angles[ix, iy] - angles[ix, iy_plus]

        #take the cosine of the angle differences
        cosdiff_next = cos(anglediff_next)
        cosdiff_prev = cos(anglediff_prev)
        cosdiff_left = cos(anglediff_left)
        cosdiff_right = cos(anglediff_right)

        # Perform operations directly on the memory view slice, basing off old_new too
        current_energy[old_new, ix, iy] = (0.5 - 1.5 * pow(cosdiff_next, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_prev, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_left, 2)
                                           + 0.5 - 1.5 * pow(cosdiff_right, 2))


cdef void MC_update(cnp.ndarray[cnp.float64_t, ndim=2] angles, double[:, :] rand_perturb, double[:, :, :] current_energy, double Ts, int nmax, double[:] accept_ratio, int it):
    cdef int ix, iy, total_accept = 0
    cdef double energy_diff, metrop_factor, accept

    for ix in range(nmax):
            # Compute old energy for the row
            #line_energy(angles, current_energy[0, ix, :], ix, nmax)
            line_energy(angles, current_energy, ix, nmax, 0)  
            
            # Perturb the angles randomly for the row
            for iy in range(nmax):
                angles[ix, iy] += rand_perturb[ix, iy]

            # Compute new energy for the row
            #line_energy(angles, current_energy[1, ix, :], ix, nmax)
            line_energy(angles, current_energy, ix, nmax, 1)  # for new energy

            # Determine which changes to accept for the row
            for iy in range(nmax):
                energy_diff = current_energy[1, ix, iy] - current_energy[0, ix, iy]
                metrop_factor = exp(-energy_diff / Ts)
                
                # Random float between 0 and 1
                accept = rand() / RAND_MAX #faster than doing numpy stuff
                
                # Decide whether to accept the new angle based on the Metropolis criterion
                if energy_diff <= 0 or metrop_factor >= accept:
                    total_accept += 1  # This angle change is accepted
                else:
                    # Revert the angle if not accepted
                    current_energy[1, ix, iy] = current_energy[0, ix, iy]
                    angles[ix, iy] -= rand_perturb[ix, iy]

    # Record the acceptance ratio for the entire lattice outside the parallel block
    accept_ratio[it] += total_accept / (nmax * nmax)


def MC_step(cnp.ndarray[cnp.float64_t, ndim=2] angles, double Ts, int nmax, cnp.ndarray[cnp.float64_t, ndim=1] energy_array , cnp.ndarray[cnp.float64_t, ndim=3] order_array, double[:] accept_ratio, int it):
    # Generate random perturbations
    cdef double scale=0.1+Ts
    #decided not to use gen_noise_matrix as it increased the time and couldnt replicate in C manner exactly (0 -1)
    cdef cnp.ndarray[cnp.float64_t, ndim=2] rand_perturb = np.random.normal(scale=scale, size=(nmax,nmax))
    cdef double[:, :, :] current_energy = np.zeros((2, nmax, nmax))
    # Perform the MC update step
    MC_update(angles, rand_perturb, current_energy, Ts, nmax, accept_ratio, it)

    # Sum up the new current_energy to get the total energy
    energy_array[it] = np.sum(np.asarray(current_energy[1]))

    # Calculate order parameter Q
    cdef double[:, :] Qab = get_Qab(angles, nmax)
    for i in range(2):
        for j in range(2):
            order_array[it, i, j] = Qab[i,j]

    return (np.asarray(angles), np.asarray(current_energy), energy_array, order_array, np.asarray(accept_ratio))


def main(int nsteps,int nmax,double temp,int pflag):

    # Create and initialise lattice
    # Create arrays to store energy, acceptance ratio and order values
    cdef cnp.ndarray[cnp.float64_t, ndim=3] order_array = np.zeros((nsteps + 1, 2, 2))
    cdef cnp.ndarray[cnp.float64_t, ndim=1] energy_array = np.zeros(nsteps + 1)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] accept_ratio = np.zeros(nsteps + 1)
    c_np_array = np.zeros((2, nmax, nmax), dtype=np.float64)
    cdef double[:, :, :] current_energy = c_np_array
    cdef double runtime
    cdef int it
    cdef double Ts = temp

    
    #plot initial frame of lattice
    cdef cnp.ndarray[cnp.float64_t, ndim=2] angles = initdat(nmax)
    #defining angles array etc
    plotdat(angles, current_energy, pflag, nmax)
    

    # Begin doing and timing some MC steps.
    initial = openmp.omp_get_wtime()
    #cdef double initial_time = time.time()


    #completing mc_steps algorithm
    for it in range(nsteps):
        angles, current_energy, energy_array, order_array, accept_ratio = MC_step(angles, Ts, nmax, energy_array, order_array, accept_ratio, it)

    #need to make sure when to use memory views and when not to
    cdef cnp.ndarray[cnp.float64_t, ndim=1] order = get_order(order_array, nsteps)
    
    # Record the final time and compute the runtime
    #cdef double final_time = time.time()
    #runtime = final_time - initial_time

    final = openmp.omp_get_wtime()
    runtime = final-initial
    

    
    # Final outputs
    print("Program: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(nmax,nsteps,temp,order[nsteps-1],runtime))
    # Plot final frame of lattice and generate output file
    plotdat(angles, current_energy, pflag, nmax)
    # Save data to a file
    savedat(angles, nsteps, Ts, runtime, accept_ratio, energy_array, order, nmax)
#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 4:
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================