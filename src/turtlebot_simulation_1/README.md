# INSTALLATION

This small tutorial is prepared for ubuntu 24.04 and ROS Jazzy, but should work for other versions.

## Stonefish installation

The Stonefish library repository can be found here: [https://github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish)
Installation instructions for Stonefish can be resumed into:

```bash
 sudo apt install libglm-dev libsdl2-dev libfreetype6-dev #dependencies
 cd (your prefered folder for the repository to download, preferrable not inside your catkin workspace)
 git clone https://github.com/patrykcieslak/stonefish.git # clone repository
 cd stonefish
 mkdir build
 cd build
 cmake ..
 make # or make -j(any number of threads)
 sudo make install
```

## ROS Dependencies

Now its time to install all ROS related packages so you can run the Turtlebot simulation. Go inside your ROS workspace src folder and clone the following repositories


**Stonefish ros** is a Stonefish simulator with ROS2 support, the one the turtlebot simulator will depend on. 

```bash
git clone https://github.com/patrykcieslak/stonefish_ros2.git
```

**Description packages** basically have the 3D models and urdf files that describe the components necessary for the turtlebot:

```bash
git clone https://bitbucket.com/udg_cirs/kobuki_description.git # Mobile base
git clone https://bitbucket.com/udg_cirs/swiftpro_description.git # Manipulator
git clone https://bitbucket.com/udg_cirs/turtlebot_description.git # Mobile base + manipulator (whole robot)
sudo apt install ros-jazzy-realsense2-description # (Realsense camera)
```
**Turtlebot simulation** has the launch files and configurations to launch the turtlebot simulation using stonefish
```bash
git clone https://bitbucket.org/udg_cirs/turtlebot_simulation.git
```

**Build:** Now you can build your ROS workspace using colcon build.

Note: Make sure to source your workspace /install/setup.bash after installation!

## RUNNING

You will find all the simulation launch files in the turtlebot_simulation package (this package). You can launch the Swift Pro maniulator alone, the Kobuki base alone, as well as, the whole Turtlebot 2 VMS. There are multiple files with predefined environments.

## Troubleshooting

We have experienced some students not having xacro installed, though it should be installed with the desktop-full installation of ROS. To install xacro execute:

```bash
sudo apt install ros-jazzy-xacro
```
