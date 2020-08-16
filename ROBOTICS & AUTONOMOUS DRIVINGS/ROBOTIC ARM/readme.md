# 3D modelling and visualization of Robots using ROS and Gazebo

To develop any new robotic system may be mobile robot, robotic arm, aerial robot, etc., it is impotant to simulate it in the virtual environment with maximum possible real time capabilities. This includes development of ROBOT models, their kinematic and dynamic behaviours, sensors and controllers of the robot and the surrounding environment.

Gazebo simulator developed by open source Robotics Foundation offers the ability to accurately and efficiently simulate wide range of robots in complex indoor and outdoor environments. It has a robust physics engine, high-quality graphics, and convenient programmatic and graphical interfaces. Hence this project is developed using Gazebo as a start of the Robotic simulations for future projects.

ROS kinetic is used for the developement along with Linux (Ubuntu 16.04). Two different models are tried and tested using ROS and Gazebo Simulator. 

The 2 Robot models are:
1. Differential robot
2. seven DOF arm

The model is developed using URDF and XACRO and then LAUNCH files are used to run them in Gazebo evnironment.

## Project Folder structure

1. docs 
    * Project details.pdf - detailed project report covering hardware and software description
    * differential_wheeled_robot.pdf - graphical view of hierarchy of links of differential wheel robot
    * seven_dof_arm - graphical view of hierarchy of links of seven DOF robotic arm
2. src
    * robot_simulation_in_gazebo - ROS package containing all the urdf, xacro and launch files of the project
    
3. test
    * diff_mobile_robot in gazebo.png - robot in Gazebo
    * diff robot with laser scanner view in gazebo.png - robot with laser scanner in Gazebo
    * seven dof arm in Gazebo.png - robot in Gazebo

