# Installation instructions

To create a conda environment for the project, use:

```shell
$ conda env create -f environment.yml
```

Then, activate the environment with:

```shell
$ conda activate sad
```

Then install the module with

```shell
$ pip install -e .
```

# Usage Instructions

First, the user must set the **gpu** variable to the desired value (True or False) at the head of all four files: **main.py**, **simulate.py**, **mfs_functions.py**, and **particular_functions.py**. **gpu** is **False** by default.

## Simulation Setup and Execution

Simulation parameters are chosen in **main.ipynb**. After choosing parameters, the simulation can be initialized and run by running **main.ipynb**.

### Plotting Parameters
1. `plotting`: this dictates how often to save simulation data. Setting `plotting` = 0 will not save data. Setting `plotting` = n will save every n steps, always including the first and last step.

### Grid Parameters
1. `dx`:  Grid spacing in the x axis. Note that `dz` is not user-specified; it is calculated based on other grid parameters.
2. `Lx`: x-axis domain size.
3. `Lz`: z-axis domain size.
4. `stop_time`: What time to stop the simulation.
5. `nsteps_per_second`: How many timesteps per second. `dt` will be roughly 1/`nsteps_per_second`.


### Diffusion Parameters
1. `D`: Diffusion coefficient.

### Timestepping parameters
1. `gamma`: Parameter for the Second-order IMEX scheme. See https://doi.org/10.1137/0732037. 
2. `c`: Parameter for the Second-order IMEX scheme.

### Initial Condition Parameters
1. `source_location`: (x,z) coordinates for the emission source.
2. `source_spread`: Standard deviation for the (gaussian) emission.
3. `source_type`: Either 'puff' or 'plume'. Puff will simulate a single puff starting at time 0, Plume will simulate a continuous plume of methane.
4. `n_copies`: How many copies of the initial state should be added in both positve and negative x direction. These copies are needed to satisfy the periodic boundary conditions.

### Obstacle Parameters
1. `obstacle`: True or False. Choose whether or not to simulate an obstacle on the domain.
2. `shape_params`: Currently only supported for circular obstacles. (obstacle x position, obstacle y position, obstacle radius).

### Wind MFS Parameters
1. `uinf`: Background wind speed (in x direction).
2. `rs_wind`: Radius of source points used in MFS calculation (for exterior solution).
3. `rs_wind_int`: Radius of source points used in MFS calculation (for interior solution).
4. `Ns_wind`: Number of source points to use in MFS calculation.
5. `Nb_wind`: Number of boundary points to use in MFS calculation.

### Concentration MFS Parameters
1. `rs_conc`: Radius of source points to use in MFS calculation (for exterior solution).
2. `cutoff`: When evaluating the modified bessel function of the second kind of order 0, values above `cutoff` will be set to 0.
3. `rs_conc_int`: Radius of source points to use in MFS calculation (for interior solution).
4. `Nb_conc`: Number of boundary points to use in MFS calculation.
5. `Ns_conc`: Number of source points to use in MFS calculation.
6. `Ns_conc_int`: Number of source points to use in MFS calculation (for interior solution).
7. `sigma`: Variance parameter to be used in MFS calculation (for interior solution).

## Plotting and Visualization

Functions for plotting the simulation results directly, and making gifs of the simulation, are found in **plotting.py**. The timestamp associated with the desired simulation must be set. The timestamp can be copied from the last output of **main.py**. After parameters have been chosen, plots can be displayed directly, and gifs will be saved in the **gifs** folder, after running **plotting.py** file.

# Additional Information

Code developed at the AMS department at the Colorado School of Mines. Contact at **andres_pruet@mines.edu**.