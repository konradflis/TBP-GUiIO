# Three-body-problem
Three-body problem Python simulation app, based on swarm optimization algorithms. 

> **Caution:** This project is still in development. More specific documentation will be added in the future. New functionalities are on their way as well!

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction
Initial Orbit Determination (IOD) is a process of establishing a body's orbit properties with a limited number of measurements in time. An object, like a satellite,
can be monitored and controlled based on the sparse observations of its state (positions and/or velocities). With just a few measurements and assumptions on the gravitational
forces, one can estimate the initial state and further propagate it in time to get complete information about the body's predicted movement.

There are many different approaches to the problem, and their usage depends on the available data. In my model, the expected orbits' positions and velocities in time are known,
as they come from [Three Body Periodic Orbits](https://ssd.jpl.nasa.gov/tools/periodic_orbits.html). In such a case, there is no need to include an additional measurement mechanism
or to change the units from metric to angular - states and velocities are given in [km] and [km/s] respectively. In a real-life scenario, it would be necessary to include the
additional radar or tracking satellite, to estimate the angles and measurement errors.

The gravitational forces can be included if assumptions on the bodies affecting the satellite are made. This simulation uses the Circular Restricted Three-Body Problem (CR3BP) 
movement equations. Both Moon and Earth determine the motion of the satellite. center of mass is the initial reference point.

This app uses three swarm optimization algorithms to find the initial states:
- Particle Swarm Optimization,
- Two-stage Particle Swarm Optimization (my own approach, using the original algorithm),
- Artificial Bee Colony Algorithm.
  
The choice was based on the model's and objective function's properties. This six-dimensional space is full of local minimums and reacts abruptly to even minor changes in the initial conditions.
The algorithms are expected to both explore the space and exploit the most promising initial states, to be able to determine the initial state and the initial orbit itself.

## Installation
The requirements are gathered below. You can find them in 'requirements.txt'.
| Library/Tool        | Version   | Notes                                     |
|---------------------|-----------|-------------------------------------------|
| Python              | 3.11+     | Algorithms and GUI were created in Python |
| PyQt6               | 6.7.1     | GUI                                       |
| matplotlib          | 3.9.2     | Results visualisation                     |
| numpy               | 1.26.4    | Calculations                              |
| scipy               | 1.13.1    | Differential equations                    |
| pandas              | 2.2.2     | Loading and converting .csv data          |
| pytest              | 7.4.4     | Unit tests                                |
| pytest-cov          | 6.0.0     | Unit tests - coverage                     |
| pytest-mock         | 3.14.0    | Unit tests - mocking                      |
| pytest-qt           | 4.2.0     | Unit tests - PyQt elements                |

## Usage
To be updated

## License
To be updated

### Third-Party Licenses
The project includes code and data from third-party sources:
- Combinear by DevSec Studio, licensed under MIT License. See [LICENCES/Combinear_license.txt](LICENCES/Combinear_license.txt) for details.
- JPL Three-Body Periodic Orbit Catalog data. See [Three Body Periodic Orbits](https://ssd.jpl.nasa.gov/tools/periodic_orbits.html) for details.

