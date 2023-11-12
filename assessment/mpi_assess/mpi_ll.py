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
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpi4py import MPI



#=======================================================================
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
def plotdat(arr,pflag,nmax):
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
    u = np.cos(arr)
    v = np.sin(arr)
    x = np.arange(nmax)
    y = np.arange(nmax)
    cols = np.zeros((nmax,nmax))
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
    plt.show()
#=======================================================================
def savedat(arr,nsteps,Ts,runtime,ratio,energy,order,nmax):
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
"""
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
#=======================================================================

#vectorised approach to one_energy for speedup:
def one_energy(arr, ix, iy, nmax):
    # Calculate neighbor indices with wraparound
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    # Calculate differences with neighbors
    xp_next = arr[ix, iy] - arr[ixp, iy]
    xm_next = arr[ix, iy] - arr[ixm, iy]
    yp_next = arr[ix, iy] - arr[ix, iyp]
    ym_next = arr[ix, iy] - arr[ix, iym]

    # Calculate energy
    en = 0.5 * (1.0 - 3.0 * np.cos(xp_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(xm_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(yp_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(ym_next)**2)

    return en

"""
def all_energy(arr,nmax):
    
    enall = 0.0
    for i in range(nmax):
        for j in range(nmax):
            enall += one_energy(arr,i,j,nmax)
    return enall
"""

#vectorised approach to all_energy, used the same one_energy function calculation and eliminated the for loops.
def all_energy(arr, nmax):
    # Calculate neighbor indices with wraparound for the entire array
    ix, iy = np.indices(arr.shape)
    ixp = (ix + 1) % nmax
    ixm = (ix - 1) % nmax
    iyp = (iy + 1) % nmax
    iym = (iy - 1) % nmax

    # Calculate differences with neighbors
    xp_next = arr[ix, iy] - arr[ixp, iy]
    xm_next = arr[ix, iy] - arr[ixm, iy]
    yp_next = arr[ix, iy] - arr[ix, iyp]
    ym_next = arr[ix, iy] - arr[ix, iym]

    # Calculate energy contributions for each cell
    en = 0.5 * (1.0 - 3.0 * np.cos(xp_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(xm_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(yp_next)**2) + \
         0.5 * (1.0 - 3.0 * np.cos(ym_next)**2)

    # Sum up all energies to get total energy
    enall = np.sum(en)

    return enall
#=======================================================================

#vectorised version of get_order for speed up
def get_order(arr, nmax):
    # Create the 3D unit vector for each cell (i,j)
    lab = np.vstack((np.cos(arr), np.sin(arr), np.zeros_like(arr))).reshape(3, nmax, nmax)
    
    # Calculate the Q tensor using einsum to replace the nested loops
    Qab = 3.0 * np.einsum('aij,bij->ab', lab, lab) / (2.0 * nmax * nmax)
    
    # Adjust the subtraction of the delta part to match the original function's logic
    delta = np.eye(3)
    Qab -= delta / (2.0 * nmax * nmax) * nmax * nmax

    # Calculate and return the maximum eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(Qab)
    return eigenvalues.max()


"""
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
def MC_step(arr, Ts, nmax, chunk_size):
    # Scale factor for angle change
    scale = 0.1 + Ts
    #counter to track the number of accepted changes
    accept = 0

    # Generate arrays of random integers for x and y indices within the lattice range
    xran = np.random.randint(0, high=nmax, size=(chunk_size, nmax))
    yran = np.random.randint(0, high=nmax, size=(chunk_size, nmax))
    # Generate an array of normally distributed random numbers for angle changes
    aran = np.random.normal(scale=scale, size=(chunk_size, nmax))

    # Iterate over chunk of the entire lattice
    for i in range(chunk_size):
        for j in range(nmax):
            # Select random indices and angle change value 
            ix = xran[i, j]
            iy = yran[i, j]
            ang = aran[i, j]

            # Calculate initial energy at the selected lattice site
            en0 = one_energy(arr, ix, iy, nmax)
            # Modify the value at the lattice site by the angle change
            arr[ix, iy] += ang
            # Calculate the energy after modification
            en1 = one_energy(arr, ix, iy, nmax)

            # Accept the change if the energy is reduced or according to the boltz
            if en1 <= en0 or np.exp(-(en1 - en0) / Ts) >= np.random.uniform():
                accept += 1
            else:
                # Revert the change if not accepted
                arr[ix, iy] -= ang

    # Return the number of accepted changes
    return accept


#=======================================================================

def main(program, nsteps, nmax, temp, pflag):
    """
    Arguments:
	  program (string) = the name of the program;
	  nsteps (int) = number of Monte Carlo steps (MCS) to perform;
      nmax (int) = side length of square angles to simulate;
	  temp (float) = reduced temperature (range 0 to 2);
	  pflag (int) = a flag to control plotting.
    Description:
      This is the main function running the Lebwohl-Lasher simulation.
    Returns:
      NULL
    """
    comm = MPI.COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    if(rank==0):
      initial = time.time()
      chunks_counter = int(nmax/size)
      division = nmax%size
      chunk_sizes = [chunks_counter + (1 if r < division else 0) for r in range(size)]
    else:
        chunk_sizes = None

    chunk_sizes = MPI.COMM_WORLD.bcast(chunk_sizes, root=0)
    chunk_size = chunk_sizes[rank]
    
    
    angles = initdat(nmax)

    if(rank ==0):
      
      plotdat(angles,pflag,nmax,temp)

    energy_array = np.zeros(nsteps+1,dtype=np.dtype)
    order_array = np.zeros(nsteps+1,dtype=np.dtype)
    ratio_array = np.zeros(nsteps+1,dtype=np.dtype)
    
    
    energy_array[0] = all_energy(angles,nmax)
    order_array[0] = get_order(angles,nmax)
    ratio_array[0] = 0.5 
    

    rank_limit = 2*(rank) + (chunk_size-1)
    
    for it in range(1,nsteps+1):
        aggreg_accept = MC_step(angles,temp, nmax,chunk_size)
        
        if rank!=0:
          MPI.COMM_WORLD.send(aggreg_accept, 0, tag = 1)

        if rank==0:
          
          accept = aggreg_accept
          for r in range(1,size):
              aggreg_accept = MPI.COMM_WORLD.recv(source=r, tag = 1)
              accept += aggreg_accept

          ratio_array[it] = accept/(nmax*nmax)
        
        order_array[it] = get_order(angles,nmax)
        energy_array[it] = all_energy(angles,nmax)
        
    
    slicer_rank = angles[2*rank:rank_limit, :]
    
    if rank != 0:
    # Send this rank's section of the angles to rank 0
        slicer_rank = angles[:chunk_size, :]
        MPI.COMM_WORLD.Send([slicer_rank, MPI.FLOAT], dest=0, tag=2)

    if rank == 0:
        # Rank 0 combines all the angles segments
        start_index = chunk_size
        for r in range(1, size):
            recv_chunk_size = nmax // size + (1 if r < division else 0)
            MPI.COMM_WORLD.Recv([angles[start_index:start_index + recv_chunk_size, :], MPI.FLOAT], source=r, tag=2)
            start_index += recv_chunk_size
      
        final = time.time()
        runtime = final-initial
    
        
        print("{}: Size: {:d}, Steps: {:d}, T*: {:5.3f}: Order: {:5.3f}, Time: {:8.6f} s".format(program, nmax,nsteps,temp,order_array[nsteps-1],runtime))
        
        savedat(angles,nsteps,temp,runtime,ratio_array,energy_array,order_array,nmax)
        plotdat(angles,pflag,nmax,temp)

        average_order = sum(order_array) / nsteps
        print(average_order)


#=======================================================================
# Main part of program, getting command line arguments and calling
# main simulation function.
#
if __name__ == '__main__':
    if int(len(sys.argv)) == 5:
        PROGNAME = sys.argv[0]
        ITERATIONS = int(sys.argv[1])
        SIZE = int(sys.argv[2])
        TEMPERATURE = float(sys.argv[3])
        PLOTFLAG = int(sys.argv[4])
        main(PROGNAME, ITERATIONS, SIZE, TEMPERATURE, PLOTFLAG)
    else:
        print("Usage: python {} <ITERATIONS> <SIZE> <TEMPERATURE> <PLOTFLAG>".format(sys.argv[0]))
#=======================================================================
