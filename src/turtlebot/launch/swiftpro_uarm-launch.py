import os, yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.descriptions import ParameterValue
from launch_ros.actions import PushRosNamespace

from ament_index_python.packages import get_package_share_directory
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():

    robot_name_arg = DeclareLaunchArgument(
        'turtlebot_name',
        default_value='turtlebot',
        description='Name of the robot'
    )

    # swfitpro uarm
    swiftpro_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('swiftpro'),
                'launch',
                'swiftpro_standalone.launch.py'
            ])
        )
    )

    return LaunchDescription([
        robot_name_arg,
        PushRosNamespace(LaunchConfiguration('turtlebot_name')),
        swiftpro_launch,
    ])
