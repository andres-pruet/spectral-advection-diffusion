# Installation instructions

To create a conda environment for the project, use:

```shell
$ conda env create -f environment.yml
```

Then, activate the environment with:

```shell
$ conda activate sad
```

# Usage Instructions

First, the user must set the **gpu** variable to the desired value (True or False) at the head of all four files: **main.ipynb**, **simulate.py**, **mfs_functions.py**, and **particular_functions.py**. **gpu** is **False** by default.

## Simulation Setup and Execution

Simulation parameters are chosen in the second block of **main.ipynb**. After choosing parameters, the simulation can be initialized and run by running all blocks. in **main.ipynb**.

## Plotting and Visualization

Functions for plotting the simulation results directly, and making gifs of the simulation, are found in **plotting.ipynb**. The timestamp associated with the desired simulation must be set in the second block. The timestamp can be copied from the last output of **main.ipynb**. After parameters have been chosen, plots can be displayed directly, and gifs will be saved in the **gifs** folder, after running all blocks in the **plotting.ipynb** file.

# Additional Information

Code developed at the AMS department at the Colorado School of Mines. Contact at **andres_pruet@mines.edu**.