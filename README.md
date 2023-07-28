# isaacgym-robotics-usage
This repo provides minimal hands-on code for Isaac-Gym Robotics Algorithms.
(This repository only covers the Isaac-gym simulation. Realworld is not included)

Isaac-Gym related code is employed from following repos: 

* [Isaac-Gym] https://github.com/NVIDIA-Omniverse/IsaacGymEnvs

## Prerequisites

This repo is tested on following environment:

* Ubuntu: 20.04
* Python: 3.8.10
* isaacgym: 1.0rc4
* isaacgymenvs: 1.3.4

### Install dependencies
1. Download **Isaac Gym Preview 4 & IsaacGymEnvs**
Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then follow the installation instructions in the documentation. We highly recommend using a conda environment to simplify set up.

2. Setup Issac-gym Engine
   Goto the below directory of your computer.
    ```bash
    isaacgym
        /{to-your-path}/IsaacGym_Preview_4_Package/isaacgym/python/setup.py
    isaacgymenvs
        /{to-your-path}/IsaacGymEnvs/setup.py
    ```
    And then, run below terminal line.
    ```bash
    pip3 install -e .
    ```

### Descriptions
Below is a list of files and their descriptions:

* Kinematic Solver
    1. Solve inverse kinematics in various method with 
        * [General]
        * [Augmented]
        * [Nullspace-projection]
        * [Repelling]
        * [RRT*]
        
        
* Trajectory Planning method
    1. [Task space planning]
        * [Quintic]
        * [Minimum Jerk]
        * [Linear movement]
    2. [Velocity profile method]: 
        * [Trapezoidal]
        * [S-spline method]
    
* Mobile Planning method
    1. [Differential Drive Kinematics]
    2. [Global planner]
        * [A*]
        * [RRT]
        * [RRT*]
    3. [Local planner]
        * [Pure-pursuit]
        
        
* Point-cloud
    1. [Point-cloud Projection]
    2. [RANSAC]
    3. [Iterative Closet Point]
    4. [Extrinsic calibration]
    
* Miscellaneous
    * TO-DO ...