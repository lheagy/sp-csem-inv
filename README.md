# sp-csem-inv

Code to run inversions for the coupling coefficient in an EM-SP simulation. 

The forward simulation solves the frequency-domain Maxwell's equations for a source term of the form 

$$
\vec{J}_{sp} = - \frac{L}{\rho g} \nabla p
$$

Where 
- $L$ is the coupling coefficient
- $rho$ is the density of water
- $g$ is the gravitational acceleration
- $p$ is the pore pressure

The inversion is set up to invert for the coupling coefficient $L$ 

## Contents 
The code is built on [SimPEG](https://simpeg.xyz), and the main files that are needed are in the [spcsem](./spcsem) directory. The [e3d](./e3d) directory contains files used in the E3D simulations, including the mesh, conductivity model, data points, and Jsp source vector. The [notebooks](./notebooks) directory contains several example notebooks. 

## Installing

From a command line, you can clone this repository by running 
```
git clone https://github.com/lheagy/sp-csem-inv.git
```

You will then need to `cd` into the `sp-csem-inv` directory and build the environment, 
```
cd sp-csem-inv
conda env create -f environment.yml
```
Note that if you are running on a Mac with an M1/M2 chip (or other ARM architecture), you will want comment out the installation of `pydiso` and instead install `python-mumps` as the solver. 

From there, you can launch jupyter and run the notebooks 
```
jupyter lab
```

## License
This code is provided under the [MIT license](./LICENSE)
